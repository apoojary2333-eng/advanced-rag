import os
from openai import OpenAI
from langchain_core.documents import Document

class RAGGenerator:
    SYSTEM_PROMPT = """You are a helpful assistant. Answer ONLY using the provided context.
For every claim you make, cite the source using [Source N] notation.
If the context does not contain the answer, say "I don't have enough information."
Never make up facts."""

    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("UPSTAGE_API_KEY"),
            base_url="https://api.upstage.ai/v1",
        )

    def _build_context(self, docs: list[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", f"Chunk {i}")
            parts.append(f"[Source {i+1}] ({source})\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def generate(self, query: str, docs: list[Document]) -> dict:
        context = self._build_context(docs)
        response = self.client.chat.completions.create(
            model="solar-pro",
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            temperature=0.1,
            max_tokens=800,
        )
        answer = response.choices[0].message.content
        return {
            "answer": answer,
            "sources": [d.metadata.get("source", "unknown") for d in docs],
            "num_sources": len(docs),
        }
