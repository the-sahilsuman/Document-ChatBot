from langchain_core.runnables import RunnableLambda


def context(docs):

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return context

create_context=RunnableLambda(context)

