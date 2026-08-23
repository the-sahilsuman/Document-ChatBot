from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embedding_model = HuggingFaceEndpointEmbeddings(
    model=os.getenv("HUGGINGFACE_EMBEDDING_MODEL")
)
