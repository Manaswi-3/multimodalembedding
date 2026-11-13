import json
import logging
import mimetypes
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TypedDict
from urllib.parse import parse_qs

import torch
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR / "images"
WEB_DIR = PROJECT_DIR / "web"
RELEVANCE_THRESHOLD = 0.6


class RetrievedImage(TypedDict):
    filename: str
    score: float


class AskResult(TypedDict):
    answer: str
    retrieved: list[RetrievedImage]


# ---------- 1. LangChain embeddings wrapper ----------
class EmbeddingGemma2Embeddings(Embeddings):
    def __init__(self, model_name="google/embeddinggemma-2", truncate_dim=None):
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float32  # never float16
        )
        self.model = SentenceTransformer(
            model_name,
            model_kwargs={"torch_dtype": dtype},
            config_kwargs={"audio_config": None},  # text + image only
        )
        self.kw = {"normalize_embeddings": True}
        if truncate_dim:
            self.kw["truncate_dim"] = truncate_dim

    # text documents
    def embed_documents(self, texts):
        return self.model.encode(texts, prompt_name="Document", **self.kw).tolist()

    # text queries
    def embed_query(self, text):
        return self.model.encode(text, prompt_name="SearchQuery", **self.kw).tolist()

    # images (called by Chroma.add_images); no prompt prefix for images
    def embed_image(self, uris):
        inputs = [{"text": "<|image|>", "image": u} for u in uris]
        return self.model.encode(inputs, **self.kw).tolist()


# ---------- 2. Chroma vector store ----------
embeddings = EmbeddingGemma2Embeddings()

vectorstore = Chroma(
    collection_name="image_rag",
    embedding_function=embeddings,
    persist_directory=str(PROJECT_DIR / "chroma_db"),
    collection_metadata={"hnsw:space": "cosine"},
)


# ---------- 3. Index images ----------
def index_images(folder: str):
    image_folder = Path(folder)
    if not image_folder.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_folder}")

    paths = sorted(
        str(p) for p in image_folder.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not paths:
        logging.warning("No supported images found in %s", image_folder)
        return

    vectorstore.add_images(
        uris=paths,
        metadatas=[{"path": p} for p in paths],
        ids=paths,  # same path = overwritten, so re-running doesn't duplicate
    )
    print(f"Indexed {len(paths)} images")


# ---------- 4. Retrieve + generate ----------
llm = ChatOllama(model="gemma3:4b", temperature=0.2)


def ask(question: str, k: int = 3) -> AskResult:
    matches = [
        (document, score)
        for document, score in vectorstore.similarity_search_with_relevance_scores(question, k=k)
        if score > RELEVANCE_THRESHOLD
    ]
    if not matches:
        return {
            "answer": f"No images scored above the relevance threshold of {RELEVANCE_THRESHOLD:.1f}.",
            "retrieved": [],
        }

    # For images, Chroma's page_content is the base64-encoded image
    content = [{
        "type": "text",
        "text": (
            "Answer the question by carefully inspecting the attached images. "
            "The images were selected by semantic search and may be irrelevant "
            "or only partially relevant; their retrieval does not prove that "
            "they contain the answer. Base factual claims only on information "
            "that is actually visible and readable in the images. For tables "
            "or datasets, inspect the visible entries before making claims; "
            "do not assume a requested record is present, and do not claim it "
            "is absent from an entire dataset if the image is incomplete, "
            "unclear, or does not show all records. If the answer cannot be "
            "verified from the images, say so plainly and describe what is "
            "uncertain. Do not guess or invent names, rows, or details.\n\n"
            f"Question: {question}"
        ),
    }]
    retrieved = []
    for d, score in matches:
        filename = Path(d.metadata.get("path", "")).name
        content.append({
            "type": "text",
            "text": f"Retrieved image: {filename}",
        })
        mime = mimetypes.guess_type(d.metadata.get("path", ""))[0] or "image/jpeg"
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{d.page_content}"},
        })
        retrieved.append({
            "filename": filename,
            "score": float(score),
        })

    print("Retrieved:", retrieved)
    return {
        "answer": llm.invoke([HumanMessage(content=content)]).content,
        "retrieved": retrieved,
    }


class QueryHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            page = (WEB_DIR / "index.html").read_bytes()
            self._send(200, "text/html; charset=utf-8", page)
        elif self.path == "/style.css":
            stylesheet = (WEB_DIR / "style.css").read_bytes()
            self._send(200, "text/css; charset=utf-8", stylesheet)
        else:
            self._send(404, "text/plain; charset=utf-8", b"Not found")

    def do_POST(self):
        if self.path != "/ask":
            self._send(404, "application/json; charset=utf-8", b'{"error":"Not found"}')
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length > 8192:
            self._send(413, "application/json; charset=utf-8", b'{"error":"Input is too large"}')
            return

        encoded_body = self.rfile.read(content_length).decode("utf-8")
        question = parse_qs(encoded_body).get("question", [""])[0].strip()
        if not question:
            self._send(400, "application/json; charset=utf-8", b'{"error":"Enter a question"}')
            return

        try:
            result = ask(question)
        except Exception:
            logging.exception("Failed to answer query")
            response = json.dumps({"error": "The query could not be completed. Check the server logs."})
            self._send(500, "application/json; charset=utf-8", response.encode("utf-8"))
            return

        response = json.dumps(result)
        self._send(200, "application/json; charset=utf-8", response.encode("utf-8"))

    def log_message(self, format, *args):
        logging.info("%s - %s", self.address_string(), format % args)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    index_images(str(IMAGE_DIR))
    server = ThreadingHTTPServer(("127.0.0.1", 8000), QueryHandler)
    logging.info("Open http://127.0.0.1:8000 to ask about your images")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down")
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
