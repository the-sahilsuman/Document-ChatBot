# eval_retriever.py
from dotenv import load_dotenv
import os
import json

load_dotenv()

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from rag.retrieval import retriever
from rag.aws_model import aws_judge_model
from rag.gemini_model import google_judge_model
# from rag.reranker import RerankingRetriever




GOLDEN_PATH = "golden_datasets/retriever_dataset.json"
JUDGE_MODEL = google_judge_model  # NOTE: differs from the other evals (gpt-4o-mini)
THRESHOLD = 0.7

with open(GOLDEN_PATH) as f:
    goldens=json.load(f)


# rank_retriever=RerankingRetriever()


def run(retriever):

    test_cases = []
    
    for g in goldens[:5]:
        retrieved = retriever.invoke(g["query"])
        retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    # 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True,async_mode=False),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True,async_mode=False)
    ]

    # 4. EVALUATE --- every metric on every case, batched + parallel, printed report.
    #    hyperparameters travel with the run so the report is tagged with the config.
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": "reranker",          # vs "reranked" when you swap it in
            "embedding_model": os.getenv("HUGGINGFACE_EMBEDDING_MODEL"),
            "chunk_size": 200,
            "chunk_overlap": 10,
            "top_k": 5,
            "judge_model": os.getenv("HUGGINGFACE_JUDGE_MODEL"),
            "golden_set": GOLDEN_PATH,
        }
    )
    return result


def run_local():
    """Standalone convenience: build the retriever, then run."""
    return run(retriever)


if __name__ == "__main__":
    # run_local()
    print("retriever", run_local())