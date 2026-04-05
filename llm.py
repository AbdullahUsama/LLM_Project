from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

SYSTEM_PROMPT = """You are a helpful customer support assistant for NUST Bank.

Your job is to answer questions about NUST Bank products, services, account features, eligibility, fees, limits, procedures, mobile banking, internet banking, transfers, and related FAQs.

Rules:
- Answer only from the provided context.
- If the question is outside NUST Bank product or app scope, say you can only help with NUST Bank product and app questions.
- If the context does not contain enough information, say: "I don't have enough information in the provided knowledge base to answer that accurately."
- Do not guess, infer, or invent details.
- Do not add policies, fees, limits, eligibility rules, or procedures unless they are explicitly in the context.
- Preserve exact product names, numbers, percentages, limits, dates, and conditions from the context.
- If multiple context chunks conflict, mention the conflict and avoid guessing.
- Keep the answer concise, factual, and directly responsive.
- If the question is ambiguous, ask one short clarifying question.
- Return only the final answer.
- Do not reveal drafts, internal reasoning, self-corrections, or meta commentary.
- Do not say things like "Wait", "Draft", "Revised Draft", "Final Decision", or "I will check".
"""

PROMPT_TEMPLATE = PromptTemplate.from_template(
    f"""System:
{SYSTEM_PROMPT}

Context:
{{context}}

Question:
{{question}}

Answer:"""
)

llm = OllamaLLM(
    model="qwen3.5:4b",
    keep_alive="30m",
    num_predict=128,
    temperature=0,
    top_p=1,
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vec_db = FAISS.load_local(
    "data/vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vec_db.as_retriever(search_kwargs={"k": 2}),
    chain_type_kwargs={
        "prompt": PROMPT_TEMPLATE,
    },
    verbose=True
)

res = qa_chain.invoke({"query": "how do i open an account?"})
print("response:", res)