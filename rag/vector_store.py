from langchain_chroma import Chroma

from rag.huggingface_model import embedding_model
from rag.text_splitter import create_chunks

vector_store=Chroma(
        embedding_function=embedding_model,
        persist_directory="chroma_db",
        collection_name="sample"
    )

def create_vector_store():

    chunks=create_chunks()

    vector_store.add_documents(chunks)

    print("Documents embedded and stored successfully.")
    return vector_store


def load_vector_store():
    return vector_store


def create_retriever():

    vector_store=load_vector_store()
    
    retriever=vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 1
        }
    )

    return retriever
