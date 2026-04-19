from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

class CrossEncoderReranker:
    MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self):
        print("Loading cross-encoder (first run downloads ~80 MB)...")
        self.model = CrossEncoder(self.MODEL)

    def rerank(self, query: str, docs: list[Document], top_k: int = 5) -> list[Document]:
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top = [doc for doc, _ in ranked[:top_k]]
        print(f"Reranked {len(docs)} -> kept top {top_k}")
        return top
