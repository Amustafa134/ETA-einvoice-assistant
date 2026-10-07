"""
Runs RAGAS evaluation against the 15 ground-truth Q&A pairs.

This is the whole point of Phase 1: turning "I built a RAG system" into
"I measured it and here are the numbers" -- the single most common gap
flagged across junior AI engineer portfolios.

Metrics used:
    - faithfulness: does the answer stick to what's actually in the
      retrieved context, or does it drift / hallucinate?
    - answer_relevancy: does the answer actually address the question asked?
    - context_precision: are the retrieved chunks actually relevant, or
      is the system pulling in noise?
    - context_recall: did retrieval find the information needed to answer
      correctly at all?

Judge LLM: Azure OpenAI (same deployment used for generation).
Judge embeddings: the same local multilingual-e5-base model used for
retrieval -- kept consistent rather than introducing a second embedding
model just for evaluation.
"""

import os
import sys
import types

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI

# --- Compatibility patch -------------------------------------------------
# Known bug: some ragas versions import ChatVertexAI from a langchain_community
# path that was removed/moved in modern langchain-community releases. We
# don't use Google Vertex AI at all (we're on Azure), so rather than chase
# exact version pins across two libraries, we insert a harmless stub module
# so ragas's import succeeds regardless of which versions are installed.
try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # placeholder -- never actually instantiated/used
        pass

    stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = stub
# --------------------------------------------------------------------------

from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from src.generation.rag_pipeline import answer_question

load_dotenv()

GROUND_TRUTH_CSV = "data/ground_truth_qa.csv"


def build_eval_dataset(question_column: str = "question_en") -> Dataset:
    """
    Run the full RAG pipeline on every ground-truth question and assemble
    a RAGAS-compatible dataset. question_column lets us evaluate the
    English or Arabic questions separately (run this twice to compare).
    """
    df = pd.read_csv(GROUND_TRUTH_CSV)

    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for idx, row in df.iterrows():
        question = row[question_column]
        print(f"  [{idx + 1}/{len(df)}] {question[:60]}...")

        result = answer_question(question)

        rows["question"].append(question)
        rows["answer"].append(result["answer"])
        rows["contexts"].append(result["contexts"])
        rows["ground_truth"].append(row["answer_en"])  # ground truth is always stored in English

    return Dataset.from_dict(rows)


def get_ragas_llm() -> LangchainLLMWrapper:
    azure_llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        temperature=0,
    )
    return LangchainLLMWrapper(azure_llm)


def get_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    hf_embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    return LangchainEmbeddingsWrapper(hf_embeddings)


def run_evaluation(question_column: str = "question_en"):
    print(f"Building evaluation dataset using '{question_column}'...")
    dataset = build_eval_dataset(question_column)

    print("\nRunning RAGAS evaluation (this calls the LLM judge multiple times, may take a few minutes)...")
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=get_ragas_llm(),
        embeddings=get_ragas_embeddings(),
    )

    print("\n" + "=" * 50)
    print(f"RESULTS ({question_column})")
    print("=" * 50)
    print(results)

    # Save per-question results too, not just the aggregate -- useful for
    # spotting which specific questions are weak, not just the overall score.
    results_df = results.to_pandas()
    out_path = f"evaluation_results_{question_column}.csv"
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nPer-question results saved to {out_path}")

    return results


if __name__ == "__main__":
    run_evaluation(question_column="question_en")
    