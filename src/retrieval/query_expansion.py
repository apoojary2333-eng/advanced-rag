import os
from openai import OpenAI

class QueryExpander:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("UPSTAGE_API_KEY"),
            base_url="https://api.upstage.ai/v1",
        )

    def expand_with_hyde(self, query: str) -> list[str]:
        response = self.client.chat.completions.create(
            model="solar-pro",
            messages=[
                {"role": "system", "content": "Write a short factual paragraph answering the question. Be concise."},
                {"role": "user",   "content": query}
            ],
            max_tokens=200,
            temperature=0.2,
        )
        hypothetical = response.choices[0].message.content
        rephrase = self.client.chat.completions.create(
            model="solar-pro",
            messages=[{"role": "user", "content": f"Write 2 rephrasings of this question (one per line):\n{query}"}],
            max_tokens=100,
            temperature=0.7,
        )
        rephrasings = rephrase.choices[0].message.content.strip().split("\n")
        return [query, hypothetical] + rephrasings[:2]
