from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

class HybridRetriever:
    def __init__(self, vectorstore, chunks: list[Document]):
        self.vectorstore = vectorstore
        self.chunks = chunks

        # Use stable chunk IDs instead of fragile page_content equality
        self.chunk_id_to_index: dict[int, int] = {}
        for i, c in enumerate(chunks):
            cid = c.metadata.get("chunk_id")
            if cid is not None:
                self.chunk_id_to_index[int(cid)] = i

        tokenised = [doc.page_content.lower().split() for doc in chunks]
        self.bm25 = BM25Okapi(tokenised)


    def retrieve(self, query: str, k: int = 20, alpha: float = 0.5) -> list[Document]:
        dense_results = self.vectorstore.similarity_search(query, k=k)
        bm25_scores   = self.bm25.get_scores(query.lower().split())

        rrf_scores: dict[int, float] = {}
        RRF_K = 60

        for rank, doc in enumerate(dense_results):
            cid = doc.metadata.get("chunk_id")
            idx = None
            if cid is not None:
                idx = self.chunk_id_to_index.get(int(cid))

            # If chunk_id missing/mismatched, skip dense contribution (safer than wrong mapping)
            if idx is not None:
                rrf_scores[idx] = rrf_scores.get(idx, 0) + alpha / (RRF_K + rank + 1)


        bm25_ranking = sorted(range(len(bm25_scores)),
        key=lambda i: bm25_scores[i], reverse=True)
        for rank, idx in enumerate(bm25_ranking[:k]):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (1 - alpha) / (RRF_K + rank + 1)

        top_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
        return [self.chunks[i] for i in top_indices]
