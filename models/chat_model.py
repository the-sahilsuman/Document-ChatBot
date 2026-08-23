from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm=ChatGoogleGenerativeAI(
    model=os.getenv("GOOGLE_CHAT_MODEL")
    # temperature=0
)
