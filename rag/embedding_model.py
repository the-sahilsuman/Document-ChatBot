from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os


load_dotenv()

embedding_model=GoogleGenerativeAIEmbeddings(
    model=os.getenv("GOOGLE_EMBEDDING_MODEL")
    # dimensions=512
)
