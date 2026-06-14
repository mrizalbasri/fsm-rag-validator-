from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
import os

def build_baseline_rag(documents):
    # Split dokumen
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Buat vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return retriever

def baseline_answer(query, retriever, api_key=None, return_context=False):
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([d.page_content for d in docs])

    # Panggil Gemini API
    if api_key:
        genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer briefly and factually based on context."
    response = model.generate_content(prompt)

    answer = response.text if hasattr(response, 'text') else ""

    if return_context:
        return {
            "answer": answer,
            "context": context,
            "docs": docs
        }
    return answer
