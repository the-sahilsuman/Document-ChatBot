from deepeval.models import GeminiModel
from dotenv import load_dotenv

load_dotenv()

google_judge_model = GeminiModel(
    model="gemini-3.5-flash-lite",
    temperature=0
)