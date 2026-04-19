import os
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredHTMLLoader
)
from langchain_core.documents import Document

class DocumentLoader:
    LOADERS = {
        ".pdf":  PyPDFLoader,
        ".txt":  TextLoader,
        ".html": UnstructuredHTMLLoader,
    }

    def load_directory(self, directory: str) -> list[Document]:
        docs = []
        for path in Path(directory).rglob("*"):
            loader_cls = self.LOADERS.get(path.suffix.lower())
            if loader_cls:
                try:
                    docs.extend(loader_cls(str(path)).load())
                    print(f"Loaded: {path.name}")
                except Exception as e:
                    print(f"Failed {path.name}: {e}")
        print(f"\nTotal documents loaded: {len(docs)}")
        return docs
