import os
import boto3
from botocore.config import Config
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_aws import ChatBedrock
from langchain_classic.chains import RetrievalQA

def RAG_pdf():
    # 1. Load Data
    data_load = PyPDFLoader("Company s leave policy.pdf")
    documents = data_load.load()
    
    # 2. Split Text
    # Reduced chunk_size: 10,000 is too large for most embedding models; 1000 is standard.
    pdf_split = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""], 
        chunk_size=2000, 
        chunk_overlap=200
    )
    texts = pdf_split.split_documents(documents)

    # 3. Create Embeddings with Retry Logic (to avoid ThrottlingException)
    retry_config = Config(retries={'max_attempts': 10, 'mode': 'standard'})
    bedrock_runtime = boto3.client("bedrock-runtime", config=retry_config)
    
    pdf_emb = BedrockEmbeddings(
        client=bedrock_runtime,
        model_id='amazon.titan-embed-text-v1'
    )
    
    # 4. Create FAISS Vector Store
    vectorstore = FAISS.from_documents(texts, pdf_emb)
    return vectorstore

def RAG_llm():
    return ChatBedrock(
        credentials_profile_name="default",
        model_id="amazon.nova-lite-v1:0",
        model_kwargs={
            "temperature": 0.3,
            "top_p": 0.9,
            "max_new_tokens": 400, # Updated from max_gen_len for Nova
        }
    )

def RAG_response(vectorstore, question):
    # This is the modern LangChain replacement for index.query
    llm = RAG_llm()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    # Run the chain
    result = qa_chain.invoke({"query": question})
    return result["result"]
