from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class SmartChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, documents: list[Document]) -> list[Document]:
        chunks = self.splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["char_count"] = len(chunk.page_content)
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
