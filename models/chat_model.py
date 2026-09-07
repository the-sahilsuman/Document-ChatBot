from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

from models.prompt import chat_template

llm=ChatGoogleGenerativeAI(
    model=os.getenv("GOOGLE_CHAT_MODEL")
    # temperature=0
)


if __name__=="__main__":
    context=[
    "Google Calendar can automatically scan emails in your Gmail and create events when it detects travel/booking information.",
    "I am a software developer focused on Cloud, DevOps, Backend Development, and Generative AI. ",
    "I build reliable backend systems, automate workflows, and deploy scalable applications using Python, AWS, Docker, and Kubernetes.",

    """Currently, I am deepening my expertise in FastAPI, cloud-native architecture, 
    infrastructure automation, LangChain, LLMs, and RAG, with a focus on building 
    and deploying production-ready systems.""",

    "Runnable.invoke() accepts the input as one argument. Your second positional argument is being interpreted as config, which must be a dictionary."
    ]

    query="what are his skills."
    chain= chat_template | llm
    response=chain.invoke({"context":context,"query":query})
    print(response.text)