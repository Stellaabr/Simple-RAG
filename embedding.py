import os
import gc
import torch
from typing import List


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings



from semantic_chunker import SemanticChunker

class PrefixedEmbeddings(Embeddings):
    def __init__(self, base_embeddings):
        self.base = base_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # добавляем префикс для документов
        prefixed = [f"document: {t}" for t in texts]
        return self.base.embed_documents(prefixed)

    def embed_query(self, text: str) -> List[float]:
        # добавляем префикс для запроса
        return self.base.embed_query(f"query: {text}")

torch.cuda.empty_cache()
gc.collect()

base_embeddings = HuggingFaceEmbeddings(
    model_name="Alibaba-NLP/gte-multilingual-base",
    model_kwargs={
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "trust_remote_code": True,
    },
    encode_kwargs={
        "normalize_embeddings": True,
    },
)


embeddings = PrefixedEmbeddings(base_embeddings)

semantic_chunker = SemanticChunker(
    embedding_model=embeddings,
    max_chunk_size=900,
    similarity_threshold=0.75,
)

all_chunks: List[Document] = []
folder_path = "/content/dataset/"

for root, _, files in os.walk(folder_path):
    for filename in files:
        if not (filename.endswith(".txt") or filename.endswith(".TXT")):
            continue

        filepath = os.path.join(root, filename)
        print(f"Текущий файл {filepath}")
        loader = TextLoader(filepath)
        docs = loader.load()

        for doc in docs:
            semantic_chunks = semantic_chunker.chunk_text(
                doc.page_content
            )

            for chunk_text in semantic_chunks:
                all_chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata=doc.metadata,
                    )
                )

        del docs
        gc.collect()

prepared_chunks = [
    {
        "text": chunk.page_content,
        "source": chunk.metadata.get("source", ""),
    }
    for chunk in all_chunks
]

BATCH_SIZE = 16

texts = [c["text"] for c in prepared_chunks]
metadata = [{"source": c["source"]} for c in prepared_chunks]

first_texts = texts[:BATCH_SIZE]
first_metadata = metadata[:BATCH_SIZE]

vector_store = FAISS.from_texts(
    texts=first_texts,
    embedding=embeddings,
    metadata=first_metadata,
)


for i in range(BATCH_SIZE, len(texts), BATCH_SIZE):
    batch_texts = texts[i:i + BATCH_SIZE]
    batch_metadata = metadata[i:i + BATCH_SIZE]

    vector_store.add_texts(
        texts=batch_texts,
        metadata=batch_metadata,
    )

    torch.cuda.empty_cache()
    gc.collect()

    print(f"{i + len(batch_texts)} / {len(texts)}")

vector_store.save_local("vector_store")


#vector_store.similarity_search("",k=3)
