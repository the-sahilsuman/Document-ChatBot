# eval_rag_pipeline.py
from dotenv import load_dotenv
import json
import os

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

load_dotenv()

from models.rag_pipeline import RagPipeline
from rag.gemini_model import google_judge_model

GOLDEN_PATH = "golden_datasets/retriever_dataset.json"   # reuse the queries
JUDGE_MODEL = google_judge_model
THRESHOLD = 0.7

with open(GOLDEN_PATH) as f:
    goldens=json.load(f)


def run(rag):

    # 2. RUN THE INJECTED PIPELINE per query, build a test case from LIVE output
    test_cases = []
    for g in goldens[:1]:
        result = rag.invoke(g["query"])          # retrieve -> rerank -> generate

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=result["answer"],       # what the generator produced
                retrieval_context=result["context"],  # what the RETRIEVER returned
            )
        )

    # 3. THE THREE TRIAD METRICS
    metrics = [
        ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]

    # 4. EVALUATE
    result = evaluate(test_cases=test_cases, metrics=metrics)
    return result


def run_local():
    """Standalone convenience: build the pipeline, then run."""
    return run(RagPipeline())


if __name__ == "__main__":
    print("rag_pipeline", run_local())