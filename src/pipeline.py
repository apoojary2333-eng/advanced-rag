import os
from dotenv import load_dotenv
load_dotenv()

from ingestion.loader         import DocumentLoader
from ingestion.chunker        import SmartChunker
from retrieval.vectorstore    import VectorStore
from retrieval.hybrid_search  import HybridRetriever
from retrieval.query_expansion import QueryExpander
from postretrieval.reranker   import CrossEncoderReranker
from generation.generator     import RAGGenerator
from langchain_core.documents import Document

class AdvancedRAGPipeline:

    def __init__(self):
        self.vectorstore  = VectorStore()
        self.reranker     = CrossEncoderReranker()
        self.expander     = QueryExpander()
        self.generator    = RAGGenerator()
        self.hybrid       = None
        self.chunks       = []

    def ingest(self, data_dir: str = "./data") -> None:
        loader  = DocumentLoader()
        chunker = SmartChunker(chunk_size=512, chunk_overlap=100)
        raw_docs     = loader.load_directory(data_dir)
        self.chunks  = chunker.split(raw_docs)
        self.vectorstore.build(self.chunks)
        self.hybrid = HybridRetriever(self.vectorstore.store, self.chunks)
        print("Ingestion complete.")

    def load_existing_index(self) -> None:
        self.vectorstore.load()
        all_docs = self.vectorstore.store.get()
        self.chunks = [
            Document(page_content=pc, metadata=meta)
            for pc, meta in zip(all_docs["documents"], all_docs["metadatas"])
        ]
        self.hybrid = HybridRetriever(self.vectorstore.store, self.chunks)
        print(f"Loaded existing index ({len(self.chunks)} chunks).")

    def query(self, question: str) -> dict:
        print(f"\nQuery: {question}")

        expanded_queries = self.expander.expand_with_hyde(question)
        print(f"  Expanded into {len(expanded_queries)} queries")

        all_candidates = []
        for q in expanded_queries:
            results = self.hybrid.retrieve(q, k=15, alpha=0.5)
            all_candidates.extend(results)

        seen, unique = set(), []
        for doc in all_candidates:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                unique.append(doc)

        print(f"  {len(unique)} unique candidates after dedup")

        top_docs = self.reranker.rerank(question, unique, top_k=5)
        result   = self.generator.generate(question, top_docs)
        return result


if __name__ == "__main__":
    rag = AdvancedRAGPipeline()

    if not os.path.exists("./chroma_db"):
        rag.ingest("./data")
    else:
        rag.load_existing_index()

    while True:
        question = input("\nAsk a question (q to quit): ").strip()
        if question.lower() == "q":
            break
        result = rag.query(question)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources used: {result['num_sources']}")
        for s in result['sources']:
            print(f"  - {s}")
