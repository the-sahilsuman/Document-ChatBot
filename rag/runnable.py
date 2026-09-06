from langchain_core.runnables import RunnableLambda

def runnable(func):
    return RunnableLambda(func)