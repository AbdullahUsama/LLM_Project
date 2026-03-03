from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

llm = OllamaLLM(model="qwen3.5:4b")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vec_db = FAISS.load_local(
    "data/vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vec_db.as_retriever(search_kwargs={"k": 3}),
    verbose=True  
)

res = qa_chain.invoke({"query": "how do i open an account?"})
print("response:", res)