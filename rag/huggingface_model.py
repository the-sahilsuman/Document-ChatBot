from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint
from deepeval.models import DeepEvalBaseLLM
from dotenv import load_dotenv
import os

load_dotenv()

embedding_model = HuggingFaceEndpointEmbeddings(
    model=os.getenv("HUGGINGFACE_EMBEDDING_MODEL")
)


# -------------------------
# Hugging Face LLM
# -------------------------

class HuggingFaceJudge(DeepEvalBaseLLM):

    def __init__(self):
        self.model = HuggingFaceEndpoint(
            repo_id=os.getenv("HUGGINGFACE_JUDGE_MODEL"),
            temperature=0
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt)

    async def a_generate(self, prompt: str) -> str:
        return self.model.invoke(prompt)

    def get_model_name(self):
        return os.getenv("HUGGINGFACE_JUDGE_MODEL")


judge_model = HuggingFaceJudge()


