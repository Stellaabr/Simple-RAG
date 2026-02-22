import torch
from typing import List

from langchain_core.documents import Document
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, pipeline


login()

MODEL_ID = "google/gemma-7b-it"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=False,
    temperature=0.1,
    repetition_penalty=1.2,
    return_full_text=False,
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 15,
        "lambda_mult": 0.7,
    },
)


def build_rag_prompt(question: str, docs: List[Document]) -> str:
    context = "\n\n".join(
        f"[Document {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

    return f"""Ты — помощник, который должен ответить на вопрос используя предоставленную информацию.
Вопрос:
{question}
Предоставленная информация:
{context}
Используй ТОЛЬКО информацию из документов. Отвечай максимально подробно.
Ответ:
"""


def ask_rag(question: str) -> str:
    docs = retriever.invoke(question)

    clean_docs = [Document(page_content=doc.page_content) for doc in docs]

    prompt = build_rag_prompt(question, clean_docs)
    output = llm(prompt)[0]["generated_text"]
    return output.strip()


question = input("Введите вопрос: ").strip()

with torch.no_grad():
    answer = ask_rag(question)

print("\nОтвет:")
print(answer)