from rag.embedding_model import embedding_model
from rag.text_splitter import create_chunks

def embed_chunks():

    chunks=create_chunks()

    texts = [chunk.page_content for chunk in chunks[:50]]

    batch_size = 50
    embed_vectors=[]

    for i in range(0, len(texts), batch_size):

        batch = texts[i:i + batch_size]

        embeddings = embedding_model.embed_documents(batch)
        embed_vectors.extend(embeddings)

    return embed_vectors

    # print("Chunks:", len(chunks))
    # print("Embeddings:", len(embed_vectors))
    # print("Embedding dimensions:", len(embed_vectors[0]))
    # print(embed_vectors)

def embed_query(query):

    return embedding_model.embed_query(query)

