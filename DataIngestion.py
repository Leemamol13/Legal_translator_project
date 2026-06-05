# ingest_csv.py
import os, pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from openai import OpenAI

load_dotenv()  # loads .env file
api_key = os.getenv("OPENAI_API_KEY")
project_id = os.getenv("OPENAI_PROJECT_ID")
client = OpenAI(api_key=api_key, project=project_id)
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

##medical_db = Chroma(persist_directory="medical_chroma", embedding_function=embeddings)
legal_db = Chroma(persist_directory="legal_chroma", embedding_function=embeddings)

def ingest_large_csv(file_path, domain, batch_size=500):
    df = pd.read_csv(file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    total_chunks = 0
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start+batch_size]
        texts = [" ".join([f"{col}: {row[col]}" for col in df.columns]) for _, row in batch.iterrows()]
        docs = [Document(page_content=t) for t in texts]
        chunks = splitter.split_documents(docs)
        if domain == "Legal":
            legal_db.add_documents(chunks)
        ##elif domain == "Medical":
           ## medical_db.add_documents(chunks)
        total_chunks += len(chunks)
        print(f"Batch {start//batch_size+1}: {len(chunks)} chunks")
   ## if domain == "Medical": medical_db.persist()
    if domain == "Legal": 
        legal_db.add_documents(chunks)
        total_chunks+=len(chunks)
        print(f"Batch {start//batch_size+1}: {len(chunks)}chunks")
    print(f" Finished ingestion: {total_chunks} chunks")


ingest_large_csv("C:/Users/LENOVO/Downloads/en-de_random_dataset_10k.csv", domain="Legal")
