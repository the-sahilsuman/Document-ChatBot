from rag.retrieval import retriever
from rag.create_context import create_context
from rag.vector_store import create_retriever

from models.prompt import chat_template
from models.chat_model import llm
from models.parsing import parser
from langchain_core.runnables import RunnablePassthrough,RunnableParallel,RunnableSequence

parallel_chain=RunnableParallel({
    "context": retriever | create_context,
    "query": RunnablePassthrough()
})

chain= parallel_chain | chat_template | llm | parser

