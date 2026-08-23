# install pypdfloader 
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "inputs"

# loader=PyPDFLoader("input")
loader=DirectoryLoader(
        path=str(PDF_DIR),
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

def load_documents():
    # having option lazy_load
    docs=loader.load()

    print(f"Documets loaded sucessfully. Total docs: {len(docs)}")
    return docs