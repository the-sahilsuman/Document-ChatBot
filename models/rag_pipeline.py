from rag.retrieval import retriever
from models.prompt import chat_template
from models.chat_model import llm


class RagPipeline:
    def __init__(self, fetch_k=10, top_k=5):
        # one retriever instance — loads the store + reranker model once
        # self.retriever = RerankingRetriever(fetch_k=fetch_k, top_k=top_k)
        self.chain= chat_template | llm
        

    def invoke(self, query: str) -> dict:
        # 1. RETRIEVE: over-fetch then rerank down to top_k Documents
        docs = retriever.invoke(query)

        # 2. UNPACK: generator wants list[str], the triad wants the same strings
        context = [doc.page_content for doc in docs]

        # 3. GENERATE: grounded answer from the retrieved context
        answer = self.chain.invoke({"context":context,"query":query})

        # return all three legs of the triad so the eval harness can score them
        return {
            "query": query,
            "context": context,
            "answer": answer.text,
        }


# quick manual smoke test: python -m src.rag_pipeline
if __name__ == "__main__":
    
    rag = RagPipeline()
    result = rag.invoke("How do I check whether bit i of an integer is set?")
    print("QUERY:  ", result["query"])
    print("ANSWER: ", result["answer"])
    print("\nCONTEXT CHUNKS:")
    for i, chunk in enumerate(result["context"]):
        print(f"  [{i}] {chunk[:120]}...")