import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

class VectorStore:
    PERSIST_DIR = "./chroma_db"

    def __init__(self):
        print("Loading local embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.store = None

    def build(self, chunks: list[Document]) -> None:
        self.store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.PERSIST_DIR,
        )
        print(f"Indexed {len(chunks)} chunks to {self.PERSIST_DIR}")

    def load(self) -> None:
        self.store = Chroma(
            persist_directory=self.PERSIST_DIR,
            embedding_function=self.embeddings,
        )

    def similarity_search(self, query: str, k: int = 10) -> list[Document]:
        if not self.store:
            raise RuntimeError("Call build() or load() first.")
        return self.store.similarity_search(query, k=k)
