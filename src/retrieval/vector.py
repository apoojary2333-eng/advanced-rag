import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

class VectorStore:
    PERSIST_DIR = "./chroma_db"

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",   # cheap + fast
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.store = None

    def build(self, chunks: list[Document]) -> None:
        """Index chunks into ChromaDB (persisted to disk)."""
        self.store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.PERSIST_DIR,
        )
        self.store.persist()
        print(f"Indexed {len(chunks)} chunks → {self.PERSIST_DIR}")

    def load(self) -> None:
        """Load an already-built index from disk."""
        self.store = Chroma(
            persist_directory=self.PERSIST_DIR,
            embedding_function=self.embeddings,
        )

    def similarity_search(self, query: str, k: int = 10) -> list[Document]:
        if not self.store:
            raise RuntimeError("Call build() or load() first.")
        return self.store.similarity_search_with_score(query, k=k)