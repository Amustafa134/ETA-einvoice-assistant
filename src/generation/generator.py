"""
Generation: takes a question + retrieved context chunks, produces an answer.

Uses Azure OpenAI (not plain OpenAI) -- authentication and model reference
work differently: Azure uses a resource endpoint + deployment name instead
of just a model name, so this is NOT interchangeable with a standard
OpenAI client without these changes.

Grounded generation only -- the system prompt explicitly instructs the
model to answer strictly from the provided context and say so clearly if
the answer isn't there, rather than filling gaps from general knowledge.
This matters both for trustworthiness and for RAGAS's faithfulness metric,
which measures exactly this.

Since this is a cross-lingual system (source docs are Arabic, questions
can be Arabic or English), the model is instructed to answer in whatever
language the question was asked, regardless of the context's language.
"""

import os

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()  # reads the AZURE_OPENAI_* variables from a local .env file

SYSTEM_PROMPT = """You are an assistant answering questions about Egypt's \
e-invoicing system, based only on the provided context from official ETA \
(Egyptian Tax Authority) documents.

Rules:
- Answer ONLY using information in the provided context. Do not use outside knowledge.
- If the answer is not in the context, say clearly that the documents don't cover this, \
rather than guessing.
- Answer in the SAME language the question was asked in (Arabic question -> Arabic \
answer, English question -> English answer), even though the source context is in Arabic.
- Be concise and direct.
"""


def get_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )


def generate_answer(question: str, context_chunks: list[str]) -> str:
    client = get_client()
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    context_text = "\n\n---\n\n".join(context_chunks)
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=deployment_name,  # for Azure, this is the DEPLOYMENT name, not a model name
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,  # deterministic-ish answers, better for evaluation consistency
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Sanity check: confirm all required env vars are actually set before
    # making an API call -- a missing var gives a confusing error otherwise.
    required_vars = ["AZURE_OPENAI_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT_NAME"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        print(f"Missing environment variables: {missing}")
        print("Check that your .env file exists in the repo root and has all four set.")
    else:
        fake_context = [
            "يمكن اصدار اكثر من اشعار مرتبط بفاتورة واحدة سابقة، "
            "ويجب مراعاة ألا يتخطى مجموع مبالغ تلك الإشعارات مبلغ الفاتورة السابق إصدارها."
        ]
        question = "Can more than one credit note be linked to the same invoice?"
        answer = generate_answer(question, fake_context)
        print(f"Question: {question}\n")
        print(f"Answer: {answer}")
        