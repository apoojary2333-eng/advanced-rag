from langchain_core.documents import Document

class ContextCompressor:
    def compress(self, docs: list[Document], query: str) -> list[Document]:
        compressed = []
        for doc in docs:
            sentences = doc.page_content.split(". ")
            query_words = set(query.lower().split())
            relevant = [s for s in sentences if any(w in s.lower() for w in query_words)]
            if relevant:
                doc.page_content = ". ".join(relevant)
            compressed.append(doc)
        return compressed
