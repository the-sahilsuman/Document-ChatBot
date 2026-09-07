# eval_retriever.py
from dotenv import load_dotenv
import os
import json

load_dotenv()

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

from rag.retrieval import retriever
from rag.aws_model import aws_judge_model
from rag.huggingface_model import judge_model

from models.prompt import chat_template
from models.chat_model import llm



GOLDEN_PATH = "golden_datasets/generator_dataset.json"
JUDGE_MODEL = judge_model  # NOTE: differs from the other evals (gpt-4o-mini)
THRESHOLD = 0.7

with open(GOLDEN_PATH) as f:
    goldens=json.load(f)

generator= chat_template | llm

def run(generator):

    test_cases = []
    
    for g in goldens[:2]:
        context = g["ideal_context"] 
        answer = generator.invoke({"context":context,"query":g["query"]})
        # retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=answer.text,             # the generated answer we're judging
                retrieval_context=[context],
            )
        )

    # 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
    metrics = [
        FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True,async_mode=False),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True,async_mode=False)
    ]

    # 4. EVALUATE --- every metric on every case, batched + parallel, printed report.
    #    hyperparameters travel with the run so the report is tagged with the config.
    result = evaluate(test_cases=test_cases, metrics=metrics)
    return result


def run_local():
    """Standalone convenience: built gernerator, then run."""
    return run(generator)


if __name__ == "__main__":
    # generator= chat_template | llm
    print("generator", run_local())