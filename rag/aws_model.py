from deepeval.models import AmazonBedrockModel
from dotenv import load_dotenv
import os

load_dotenv()


aws_judge_model = AmazonBedrockModel(
    model=os.getenv("AWS_BEDROCK_MODEL_NAME"),
    region=os.getenv("AWS_REGION"),
    generation_kwargs={
        "temperature": 0,
        "maxTokens": 1000,
    },
)
