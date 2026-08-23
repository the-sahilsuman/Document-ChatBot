from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", """
        You are a document-based assistant.

        Answer the user's question using the provided context.

        If the answer cannot be found in the context,
        say that the information is not available in the documents.

        Context:
        ----------------
        {context}
        ----------------

        Question:
        {query}

        Answer:
        """
    ),
    ("human", "Explain perfectly as expert the, {query}.")
])

# prompt = chat_template.invoke({
#     "domain": "science",
#     "topic": "black holes"
# })
