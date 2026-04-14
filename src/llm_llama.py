from __future__ import annotations

from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

SYSTEM_PROMPT = """You are a customer support assistant for NUST Bank accounts and any financial, digital, policy or banking information related to them.

Primary goal:
Help the user get the most useful, accurate answer possible about NUST Bank products, services, policies, and app features.

Use the provided context to answer questions about NUST Bank products, services, and app features.

Guidelines:
- Be concise, factual, and practical.
- Prefer information from the context.
- If the question is clearly unrelated to NUST Bank, say: "I can only help with NUST Bank product and app questions."
- If the context is missing key details, say: "I don't have enough information in the provided knowledge base to answer that accurately."
- If the question or context is ambiguous or uncertain, ask for clarification before answering.
- If needed, ask one brief clarifying question.
- Prefer actionable next steps over refusal-only replies.
- Do not stop mid-answer. If the answer has multiple parts, complete all parts before finishing.
- Provide the full answer requested unless the user explicitly asks for a short response.



"""

PROMPT_TEMPLATE = PromptTemplate.from_template(
    f"""System:
{SYSTEM_PROMPT}

Examples:
Q: Does the NUST app support instant transfers?
A: Based on the provided context, instant transfer support depends on the transfer type and destination bank. If you share whether it is NUST-to-NUST or interbank, I can give the exact steps and limits.

Q: What is the weather in Lahore today?
A: I can help best with NUST Bank products and app questions. If you want, I can still give general guidance.

Context:
{{context}}

Question:
{{question}}

Answer:"""
)


@lru_cache(maxsize=1)
def get_vectorstore() -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(
        "data/vectorstore",
        embeddings,
        allow_dangerous_deserialization=True,
    )


@lru_cache(maxsize=1)
def get_qa_chain() -> RetrievalQA:
    llm = OllamaLLM(
        model="llama3.2:3B",
        keep_alive="30m",
        num_predict=512,
        temperature=0.4,
        top_p=1,
    )
    vec_db = get_vectorstore()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vec_db.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": PROMPT_TEMPLATE},
        verbose=False,
    )


# Module-level objects for easy import in notebooks or API code.
vec_db = get_vectorstore()
qa_chain = get_qa_chain()


def answer_query(query: str) -> dict:
    return qa_chain.invoke({"query": query})


def refresh_rag_resources() -> None:
    get_vectorstore.cache_clear()
    get_qa_chain.cache_clear()

    global vec_db, qa_chain
    vec_db = get_vectorstore()
    qa_chain = get_qa_chain()
