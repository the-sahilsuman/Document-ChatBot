from langchain_core.runnables import RunnableLambda
from rag.runnable import runnable


@runnable
def create_context(docs):

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return context
