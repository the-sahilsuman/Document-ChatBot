from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.document_loader import load_documents

splitter=RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0
    )

def create_chunks():

    docs=load_documents()

    chunks=splitter.split_documents([doc for doc in docs])

    print(f"Chunks created sucessfully. Total chunks: {len(chunks)}")
    return chunks
