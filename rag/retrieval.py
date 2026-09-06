from rag.vector_store import create_retriever, create_vector_store
from rag.create_context import create_context
from rag.runnable import runnable

# retriever=create_retriever()

@runnable
def retriever(query):
    return create_retriever().invoke(query)


if __name__=="__main__":
    docs=retriever.invoke("what is bit manupulation.")
    
    print(len(docs))
    context=""
    for doc in docs:
        print(doc)
        print()
        # context+=doc.page_content+" "
    # print(context)