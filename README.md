# ETA E-Invoice Assistant

A bilingual (Arabic/English) RAG assistant over Egypt's official e-invoicing regulations, published by the Egyptian Tax Authority (ETA).

**Live capability:** ask a question in Arabic *or* English -- the system retrieves from the Arabic source documents either way, and answers in whichever language you asked in.

## Why this project

Most RAG portfolio projects are English-only demos over clean, well-formatted text. This one tackles a harder, more realistic problem: official Arabic government PDFs with broken font encodings, dense table-structured content, and a genuine cross-lingual retrieval requirement -- then measures whether the system actually works, rather than just demoing it.

## Architecture

```
Arabic PDFs (ETA guides)
    |
    v
OCR extraction (PyMuPDF render -> Tesseract, Arabic)
    |
    v
Boilerplate cleanup (strips repeated headers/footers)
    |
    v
Arabic-aware chunking (custom separators: question mark, comma, semicolon)
    |  + noise filter (drops OCR garbage from diagram pages)
    v
Embedding (intfloat/multilingual-e5-base, local)
    |
    v
Qdrant vector store (Docker)
    |
    v
Retrieval (cosine similarity)
    |
    v
Generation (Azure OpenAI, grounded + cross-lingual prompt)
    |
    v
FastAPI /ask endpoint (page-level source citations)
```

## Tech stack

- **Backend:** FastAPI, Pydantic, Uvicorn
- **Ingestion/OCR:** PyMuPDF, Tesseract OCR (Arabic)
- **Chunking:** LangChain Text Splitters (Arabic-aware)
- **Embeddings:** sentence-transformers (`intfloat/multilingual-e5-base`) -- local, free, strong Arabic performance
- **Vector store:** Qdrant (Docker)
- **Generation:** Azure OpenAI
- **Evaluation:** RAGAS (faithfulness, answer relevancy, context precision, context recall)
- **Testing:** pytest (15 tests -- ingestion logic + API layer)

## Key engineering decisions

| Decision | Why |
|---|---|
| OCR instead of PDF text-layer extraction | Source PDFs have broken font-to-Unicode mappings causing systematic character corruption (confirmed by sampling pages across both documents). OCR reads rendered glyphs, bypassing the broken encoding entirely. |
| Local embeddings over an API-based model | Free, no per-call cost, and multilingual-e5-base is a strong performer specifically on Arabic retrieval. |
| Noise-ratio chunk filter | OCR on diagram/flowchart pages produces fragmented, low-value text. Filtered automatically rather than manually curating pages. |
| Chunk size 500 -> 250 (evaluation-driven) | Baseline evaluation showed low context recall (0.378). Hypothesis: large chunks diluted relevance in the FAQ's dense Q&A table format. Reducing chunk size improved recall to 0.533 (+41% relative) -- confirmed, not assumed. |
| Retrieval score threshold (0.78) | Attempted fix for flat context precision. Evaluation showed **no measurable effect** -- context recall was unchanged, meaning the threshold filtered nothing for this query set. Informs the current hypothesis below. |
| Mocked API tests instead of live-service tests | API tests mock the pipeline so they run instantly, anywhere (including CI), with no Qdrant or Azure dependency. The real pipeline is already exercised for real by the RAGAS evaluation script -- the two test suites serve different purposes. |

## Evaluation

RAGAS metrics against 15 hand-built, source-verified ground-truth Q&A pairs:

| Metric | Baseline | After chunk-size fix | After threshold (no change) |
|---|---|---|---|
| Faithfulness | 0.833 | 0.865 | 0.911 |
| Answer Relevancy | 0.629 | 0.670 | 0.574 |
| Context Precision | 0.336 | 0.330 | 0.330 |
| Context Recall | 0.378 | **0.533** | 0.533 |

**Interpretation:** generation is reliably grounded (faithfulness stayed strong throughout -- the model isn't hallucinating). Retrieval was the real bottleneck. Shrinking chunk size fixed recall significantly. Context precision remains a known limitation -- see below.

## Known limitations

- **Context precision (~0.33) hasn't moved across iterations.** Working hypothesis: irrelevant content is co-located *within* the same chunks as relevant content (the FAQ document's dense table structure means multiple unrelated Q&A pairs can share a chunk even at reduced size). A retrieval-time score threshold can't fix this -- it would need structure-aware chunking that splits precisely at each Q&A boundary instead of generic character counts.
- A small number of diagram/flowchart pages produce noisy OCR output, filtered via a heuristic rather than fully resolved (doesn't affect the ground-truth Q&A set).
- No conversational memory yet -- each question is handled independently.

## Project structure

```
eta-einvoice-assistant/
|-- data/                    # source PDFs + ground_truth_qa.csv
|-- src/
|   |-- ingestion/           # pdf_loader.py, cleaner.py, chunker.py
|   |-- retrieval/           # embedder.py, vector_store.py
|   |-- generation/          # generator.py, rag_pipeline.py
|   |-- evaluation/          # evaluate.py (RAGAS)
|   `-- api/                 # app.py (FastAPI)
|-- tests/                   # pytest suite (15 tests)
|-- main.py                  # ingestion pipeline entry point
|-- pytest.ini
|-- requirements.txt
|-- LICENSE                  # MIT
`-- README.md
```

## Running it

```bash
# 1. Install dependencies (inside a virtual environment)
pip install -r requirements.txt

# 2. Start Qdrant
docker run -d -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage --name qdrant qdrant/qdrant

# 3. Set up your .env file (not committed -- see .gitignore)
# AZURE_OPENAI_KEY=...
# AZURE_OPENAI_ENDPOINT=...
# AZURE_OPENAI_API_VERSION=...
# AZURE_OPENAI_DEPLOYMENT_NAME=...

# 4. Ingest, chunk, embed, and store the source documents
python -m src.retrieval.vector_store

# 5. Start the API
uvicorn src.api.app:app --reload --port 8000
```

Then open `http://localhost:8000/docs` for an interactive UI to ask questions directly -- no separate frontend needed.

### Example request

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Can more than one credit note be linked to the same invoice?", "top_k": 5}'
```

### Running the tests

```bash
pytest -v
```

15 tests covering chunking logic, boilerplate cleanup, and the API layer (with the pipeline mocked for speed and isolation). Runs in under two minutes with no external services required.

### Running the evaluation

```bash
python -m src.evaluation.evaluate
```

Runs all 15 ground-truth questions through the full live pipeline (Qdrant + Azure OpenAI required) and prints RAGAS scores, plus saves per-question results to `evaluation_results_question_en.csv`.

## Future improvements

- Structure-aware chunking (split by Q&A boundary, not character count) -- directly targets the context precision limitation above
- Reranking step after retrieval
- CI/CD pipeline (tests are already written and ready to wire into one)
- Conversational memory for follow-up questions

## License

MIT -- see [LICENSE](LICENSE).

## Data sources

Official Egyptian Tax Authority (ETA) publications:
- Taxpayers Introductory Guide to the E-Invoicing System (Arabic)
- E-Invoicing Solution FAQ Guide (Arabic)

Both sourced directly from [eta.gov.eg](https://eta.gov.eg).
