# eta-einvoice-assistant

A cross-lingual RAG assistant over Egypt's official e-invoicing regulations.

## Overview

The source documents (Egyptian Tax Authority taxpayer guides) are in Arabic.
This assistant lets a user ask questions in **either Arabic or English**,
retrieves the relevant passage from the Arabic source material either way,
and answers in whichever language the question was asked.

## Problem

Regulatory and compliance teams in Egypt need fast, accurate answers from
dense official documents — but most are only comfortable working in one of
the two languages, while the source documents are Arabic-only.

## Data

Two official ETA documents, downloaded directly from eta.gov.eg:
- `Taxpayers-Introductory-Guide-Arabic_v2-2023.pdf` — e-invoicing system overview
- `e-Invoicing Solution FAQs.pdf` — official FAQ guide

`data/ground_truth_qa.csv` — 15 bilingual ground-truth Q&A pairs sourced
directly from the above documents, used for evaluation.

## Architecture

(to be filled in as we build: chunking strategy, embedding model, Qdrant,
retrieval, generation, evaluation loop)

## Status

🚧 In progress — repo scaffolded, ground-truth dataset seeded, source PDFs
identified. Next: ingestion pipeline.

## Evaluation

(to be filled in: RAGAS metrics, baseline numbers, before/after comparison)
