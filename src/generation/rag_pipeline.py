"""
Combines retrieval + generation into a single question-answering function.
This is the core of the actual RAG system -- reused both by the evaluation
script and the real API.
"""

from src.retrieval.vector_store import get_client, search
from src.generation.generator import generate_answer


def answer_question(question: str, top_k: int = 5) -> dict:
    """
    Full RAG pipeline: retrieve relevant chunks, then generate an answer
    grounded in them.

    Returns:
        question: the original question
        answer: the generated answer
        contexts: plain-text list of retrieved chunks (what RAGAS evaluation needs)
        sources: same chunks but with source_file/page_number attached
                 (what a real API response needs, for citations)
    """
    client = get_client()
    results = search(client, question, top_k=top_k)

    contexts = [r.payload["text"] for r in results]
    sources = [
        {
            "text": r.payload["text"],
            "source_file": r.payload["source_file"],
            "page_number": r.payload["page_number"],
            "score": r.score,
        }
        for r in results
    ]

    answer = generate_answer(question, contexts)

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "sources": sources,
    }


if __name__ == "__main__":
    # Quick manual check with a real question through the real pipeline.
    result = answer_question("Can more than one credit note be linked to the same invoice?")
    print(f"Question: {result['question']}\n")
    print(f"Answer: {result['answer']}\n")
    print(f"Retrieved {len(result['contexts'])} context chunks:")
    for s in result["sources"]:
        print(f"  - {s['source_file']} / page {s['page_number']} (score={s['score']:.3f})")
        