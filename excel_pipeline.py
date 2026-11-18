"""
Excel → Vector Embedding Pipeline
===================================
Processes .xlsx files from excel_files/ and feeds content into ChromaDB.

Strategy per sheet:
  - Table-only sheet  → extract table → Markdown → text embedding
  - Sheet with images → extract embedded image bytes → PNG → image embedding
  - Sheet with charts → re-render chart from data → PNG → image embedding
  - Mixed (data + visuals) → both a text chunk AND image chunk(s)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import mimetypes
import re
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TypedDict
from urllib.parse import parse_qs

import os as _os
import openpyxl
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless – no display needed
# Keep matplotlib cache inside the project to avoid permission warnings
_os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".mpl_cache"),
)
import matplotlib.pyplot as plt
from PIL import Image as PILImage

import torch
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_DIR  = Path(__file__).resolve().parent
EXCEL_DIR    = PROJECT_DIR / "excel_files"    # drop your .xlsx files here
WEB_DIR      = PROJECT_DIR / "web"
RENDERED_DIR = PROJECT_DIR / "rendered_sheets"  # auto-generated PNGs
MARKDOWN_DIR = PROJECT_DIR / "markdown_tables"  # extracted markdown table files
LOGS_DIR     = PROJECT_DIR / "logs"             # application log files
RENDERED_DIR.mkdir(exist_ok=True)
MARKDOWN_DIR.mkdir(exist_ok=True)
EXCEL_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

RELEVANCE_THRESHOLD = 0.5
TOP_K = 3   # number of results shown in the UI

LOG_FILE = LOGS_DIR / "pipeline.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_FILE), encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TypedDicts
# ---------------------------------------------------------------------------
class RetrievedChunk(TypedDict):
    label: str    # "filename :: sheet"
    score: float
    kind: str     # "table" | "image" | "chart"


class AskResult(TypedDict):
    answer: str
    retrieved: list[RetrievedChunk]


# ===========================================================================
# 1.  LangChain embeddings wrapper
# ===========================================================================
class EmbeddingGemma2Embeddings(Embeddings):
    def __init__(self, model_name: str = "google/embeddinggemma-2", truncate_dim=None):
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        self.model = SentenceTransformer(
            model_name,
            model_kwargs={"torch_dtype": dtype},
            config_kwargs={"audio_config": None},
        )
        self.kw: dict = {"normalize_embeddings": True}
        if truncate_dim:
            self.kw["truncate_dim"] = truncate_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, prompt_name="Document", **self.kw).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, prompt_name="SearchQuery", **self.kw).tolist()

    def embed_image(self, uris: list[str]) -> list[list[float]]:
        inputs = [{"text": "<|image|>", "image": u} for u in uris]
        return self.model.encode(inputs, **self.kw).tolist()


# ===========================================================================
# 2.  ChromaDB vector store
# ===========================================================================
embeddings = EmbeddingGemma2Embeddings()

vectorstore = Chroma(
    collection_name="excel_rag",
    embedding_function=embeddings,
    persist_directory=str(PROJECT_DIR / "chroma_db"),
    collection_metadata={"hnsw:space": "cosine"},
)


# ===========================================================================
# 3.  Excel analysis helpers
# ===========================================================================

def _sheet_has_visuals(ws) -> bool:
    return bool(ws._images or ws._charts)


def _sheet_has_data(ws) -> bool:
    for row in ws.iter_rows():
        for cell in row:
            if cell.value is not None:
                return True
    return False


def _sheet_to_dataframe(ws) -> pd.DataFrame:
    data = list(ws.values)
    if not data:
        return pd.DataFrame()
    headers = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(data[0])]
    return pd.DataFrame(data[1:], columns=headers)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    try:
        return df.to_markdown(index=False)
    except ImportError:
        cols = list(df.columns)
        sep  = "| " + " | ".join(["---"] * len(cols)) + " |"
        header = "| " + " | ".join(cols) + " |"
        rows = [
            "| " + " | ".join(str(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)
        ]
        return "\n".join([header, sep] + rows)


# ===========================================================================
# 4.  PNG renderers
# ===========================================================================

def _render_dataframe_to_png(df: pd.DataFrame, out_path: Path) -> None:
    nrows, ncols = df.shape
    fig, ax = plt.subplots(figsize=(max(6, ncols * 1.5), max(2, nrows * 0.4 + 1)))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.astype(str).values.tolist(),
        colLabels=list(df.columns),
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(ncols)))
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _extract_embedded_images(ws, out_dir: Path, prefix: str) -> list[tuple[Path, str]]:
    """Returns list of (png_path, kind) where kind is 'image' or 'chart'."""
    saved: list[tuple[Path, str]] = []

    for idx, img_obj in enumerate(ws._images):
        try:
            raw = img_obj._data()
            pil_img = PILImage.open(io.BytesIO(raw)).convert("RGB")
            out = out_dir / f"{prefix}_img_{idx}.png"
            pil_img.save(str(out), format="PNG")
            saved.append((out, "image"))
            log.info("Extracted embedded image → %s", out.name)
        except Exception as exc:
            log.warning("Could not extract image %d from '%s': %s", idx, prefix, exc)

    for idx, chart in enumerate(ws._charts):
        try:
            out = out_dir / f"{prefix}_chart_{idx}.png"
            _render_chart_to_png(chart, out)
            saved.append((out, "chart"))
        except Exception as exc:
            log.warning("Could not render chart %d from '%s': %s", idx, prefix, exc)

    return saved


def _render_chart_to_png(chart, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(getattr(chart, "title", "") or "Chart")

    plotted = False
    try:
        for series in chart.series:
            vals = []
            if hasattr(series, "val") and series.val:
                ref = series.val
                if hasattr(ref, "numRef") and ref.numRef:
                    for pt in ref.numRef.numCache.pt:
                        try:
                            vals.append(float(pt.v))
                        except (TypeError, ValueError):
                            pass
            if vals:
                label = ""
                try:
                    label = series.tx.strRef.strCache.pt[0].v
                except Exception:
                    pass
                ax.plot(vals, label=label or f"Series {len(ax.lines)}")
                plotted = True
    except Exception as exc:
        log.debug("Chart series parse error: %s", exc)

    if not plotted:
        ax.text(0.5, 0.5, "Chart (data not extractable)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="gray")
    else:
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Rendered chart PNG → %s", out_path.name)


# ===========================================================================
# 5.  Excel chunk dataclass
# ===========================================================================
class ExcelChunk(TypedDict):
    kind: str          # "text" | "image" | "chart"
    content: str       # markdown string  OR  absolute path to PNG
    source_file: str
    sheet: str
    metadata: dict


def process_excel_file(xlsx_path: Path) -> list[ExcelChunk]:
    chunks: list[ExcelChunk] = []
    wb = openpyxl.load_workbook(str(xlsx_path), data_only=True)

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        prefix = re.sub(r"[^\w]", "_", f"{xlsx_path.stem}__{sheet_name}")
        has_data    = _sheet_has_data(ws)
        has_visuals = _sheet_has_visuals(ws)

        log.info("Sheet '%s' | '%s': data=%s visuals=%s",
                 xlsx_path.name, sheet_name, has_data, has_visuals)

        # ── A. Pure table → Markdown text chunk ────────────────────────────
        if has_data and not has_visuals:
            df = _sheet_to_dataframe(ws)
            md = _dataframe_to_markdown(df)
            if md:
                md_content = f"# {xlsx_path.name} — {sheet_name}\n\n{md}\n"
                md_file = MARKDOWN_DIR / f"{prefix}.md"
                md_file.write_text(md_content, encoding="utf-8")
                log.info("Saved markdown table file → %s.md", prefix)

                chunks.append({
                    "kind": "text",
                    "content": md_content.strip(),
                    "source_file": str(xlsx_path),
                    "sheet": sheet_name,
                    "metadata": {
                        "source_file": xlsx_path.name,
                        "sheet": sheet_name,
                        "type": "table",
                        "md_path": str(md_file),
                    },
                })

        # ── B. Visual content → PNG image chunks ───────────────────────────
        if has_visuals:
            for png_path, visual_kind in _extract_embedded_images(ws, RENDERED_DIR, prefix):
                chunks.append({
                    "kind": visual_kind,
                    "content": str(png_path),
                    "source_file": str(xlsx_path),
                    "sheet": sheet_name,
                    "metadata": {
                        "source_file": xlsx_path.name,
                        "sheet": sheet_name,
                        "type": visual_kind,
                        "path": str(png_path),
                    },
                })
            # Also keep the table data alongside visuals
            if has_data:
                df = _sheet_to_dataframe(ws)
                md = _dataframe_to_markdown(df)
                if md:
                    md_content = f"# {xlsx_path.name} — {sheet_name}\n\n{md}\n"
                    md_file = MARKDOWN_DIR / f"{prefix}.md"
                    md_file.write_text(md_content, encoding="utf-8")
                    log.info("Saved markdown table file → %s.md", prefix)

                    chunks.append({
                        "kind": "text",
                        "content": md_content.strip(),
                        "source_file": str(xlsx_path),
                        "sheet": sheet_name,
                        "metadata": {
                            "source_file": xlsx_path.name,
                            "sheet": sheet_name,
                            "type": "table",
                            "md_path": str(md_file),
                        },
                    })

    wb.close()
    return chunks


# ===========================================================================
# 6.  Indexing
# ===========================================================================

def index_excel_folder(folder: Path | str) -> None:
    folder = Path(folder)
    xlsx_files = sorted(folder.glob("*.xlsx"))
    if not xlsx_files:
        log.warning("No .xlsx files found in %s", folder)
        return

    text_docs, text_ids, text_metas = [], [], []
    image_uris, image_ids, image_metas = [], [], []

    for xlsx_path in xlsx_files:
        log.info("Processing: %s", xlsx_path.name)
        try:
            chunks = process_excel_file(xlsx_path)
        except Exception as exc:
            log.error("Failed to process %s: %s", xlsx_path.name, exc)
            continue

        for chunk in chunks:
            if chunk["kind"] == "text":
                # text ID: file + sheet (one text chunk per sheet)
                doc_id = f"excel::text::{chunk['source_file']}::{chunk['sheet']}"
                text_docs.append(chunk["content"])
                text_ids.append(doc_id)
                text_metas.append(chunk["metadata"])
            else:
                # image/chart ID: use the PNG path — guaranteed unique per file
                doc_id = f"excel::img::{chunk['content']}"
                image_uris.append(chunk["content"])
                image_ids.append(doc_id)
                image_metas.append(chunk["metadata"])

    if text_docs:
        vectorstore.add_texts(texts=text_docs, metadatas=text_metas, ids=text_ids)
        log.info("Indexed %d text (table) chunks", len(text_docs))

    if image_uris:
        vectorstore.add_images(uris=image_uris, metadatas=image_metas, ids=image_ids)
        log.info("Indexed %d image/chart chunks", len(image_uris))


# ===========================================================================
# 7.  Retrieval + generation
# ===========================================================================
llm = ChatOllama(model="gemma3:4b", temperature=0.2)


def ask(question: str, k: int = TOP_K) -> AskResult:
    matches = [
        (doc, score)
        for doc, score in vectorstore.similarity_search_with_relevance_scores(question, k=k)
        if score > RELEVANCE_THRESHOLD
    ]

    if not matches:
        return {
            "answer": f"No relevant content found above the relevance threshold ({RELEVANCE_THRESHOLD:.1f}). Try rephrasing your question.",
            "retrieved": [],
        }

    system_prompt = (
        "You are a data analyst. Answer the user's question using ONLY the Excel content "
        "provided below — Markdown tables and/or rendered chart/image snapshots. "
        "If the data does not contain the answer, say so explicitly. "
        "Do not guess or invent values.\n\n"
        f"Question: {question}\n\n"
    )

    content: list[dict] = [{"type": "text", "text": system_prompt}]
    retrieved: list[RetrievedChunk] = []

    for doc, score in matches:
        meta   = doc.metadata or {}
        kind   = meta.get("type", "table")
        source = meta.get("source_file", "unknown")
        sheet  = meta.get("sheet", "")
        label  = f"{source} :: {sheet}" if sheet else source

        if kind == "table":
            content.append({
                "type": "text",
                "text": f"\n---\n**Source:** {label}\n\n{doc.page_content}",
            })
        else:
            path_meta = meta.get("path", "")
            mime = mimetypes.guess_type(path_meta)[0] or "image/png"
            content.append({"type": "text", "text": f"\n---\n**Source ({kind}):** {label}"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{doc.page_content}"},
            })

        retrieved.append({"label": label, "score": float(score), "kind": kind})

    log.info("Retrieved %d chunks: %s", len(retrieved), retrieved)
    answer = llm.invoke([HumanMessage(content=content)]).content
    return {"answer": answer, "retrieved": retrieved}


# ===========================================================================
# 8.  HTTP server
# ===========================================================================
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
            self._send(200, "text/html; charset=utf-8", (WEB_DIR / "index.html").read_bytes())
        elif self.path == "/style.css":
            self._send(200, "text/css; charset=utf-8", (WEB_DIR / "style.css").read_bytes())
        else:
            self._send(404, "text/plain; charset=utf-8", b"Not found")

    def do_POST(self):
        if self.path != "/ask":
            self._send(404, "application/json; charset=utf-8", b'{"error":"Not found"}')
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length > 8192:
            self._send(413, "application/json; charset=utf-8", b'{"error":"Input too large"}')
            return
        body = self.rfile.read(length).decode("utf-8")
        question = parse_qs(body).get("question", [""])[0].strip()
        if not question:
            self._send(400, "application/json; charset=utf-8", b'{"error":"Enter a question"}')
            return
        try:
            result = ask(question)
        except Exception:
            logging.exception("Query failed")
            self._send(500, "application/json; charset=utf-8",
                       b'{"error":"Query could not be completed. Check logs."}')
            return
        self._send(200, "application/json; charset=utf-8", json.dumps(result).encode())

    def log_message(self, fmt, *args):
        log.info("%s - %s", self.address_string(), fmt % args)


# ===========================================================================
# 9.  Entry point
# ===========================================================================
def main():
    log.info("=== Indexing Excel files from: %s ===", EXCEL_DIR)
    index_excel_folder(EXCEL_DIR)

    server = ThreadingHTTPServer(("127.0.0.1", 8000), QueryHandler)
    log.info("Server ready → open http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
