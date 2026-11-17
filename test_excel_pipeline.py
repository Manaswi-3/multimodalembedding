"""
Dry-run: processes all .xlsx in excel_files/ and prints what would be indexed.
No model loading — fast to run.
"""
import sys, types, unittest.mock as mock
from pathlib import Path

# Monkeypatch heavy deps so they don't load
fake_torch = types.ModuleType("torch")
fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False)
fake_torch.float32 = "float32"
fake_torch.bfloat16 = "bfloat16"

for mod, obj in [
    ("torch", fake_torch),
    ("sentence_transformers", types.ModuleType("sentence_transformers")),
    ("langchain_core", types.ModuleType("langchain_core")),
    ("langchain_core.embeddings", types.ModuleType("langchain_core.embeddings")),
    ("langchain_core.messages", types.ModuleType("langchain_core.messages")),
    ("langchain_chroma", types.ModuleType("langchain_chroma")),
    ("langchain_ollama", types.ModuleType("langchain_ollama")),
]:
    sys.modules[mod] = obj

sys.modules["sentence_transformers"].SentenceTransformer = mock.MagicMock()
sys.modules["langchain_core.embeddings"].Embeddings = object
sys.modules["langchain_core.messages"].HumanMessage = mock.MagicMock()
sys.modules["langchain_chroma"].Chroma = mock.MagicMock()
sys.modules["langchain_ollama"].ChatOllama = mock.MagicMock()

sys.path.insert(0, str(Path("c:/Users/kotim/Downloads/multimodalembedding")))
from excel_pipeline import process_excel_file, EXCEL_DIR

for xlsx in sorted(EXCEL_DIR.glob("*.xlsx")):
    print(f"\n{'='*65}")
    print(f"FILE: {xlsx.name}")
    chunks = process_excel_file(xlsx)
    if not chunks:
        print("  (no chunks extracted)")
    for c in chunks:
        print(f"  kind={c['kind']:6s}  sheet={c['sheet'][:20]:20s}  {c['content'][:90]!r}")

print("\n✅  Dry-run complete.")
