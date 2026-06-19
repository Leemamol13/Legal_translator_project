**Legal Translation Evaluation Pipeline**
**Overview**
This project provides an end‑to‑end pipeline for English–German legal translation with integrated quality evaluation. It combines AI translation, glossary injection, semantic retrieval, and industry‑standard metrics to ensure legally precise and measurable translations.

**Features**
Glossary Integration: Injects legal definitions and examples into translation prompts.

Context‑Aware Translation: Uses GPT‑4o‑mini for German translations guided by legal context.

Stoplist Filtering: Excludes drafting words (shall, may, hereby) to focus on substantive legal terms.

Semantic Retrieval: Retrieves up to 5 relevant bilingual examples (exact match → synonyms → semantic fallback).

Evaluation Metrics: Computes BLEU, METEOR, and COMET scores for each translation.

Batch Processing: Processes corpus in chunks for scalability and traceability.

Corpus‑Level KPIs: Reports average and standard deviation of BLEU, METEOR, COMET across the dataset.

**Project Structure**
en-de_random_dataset_10k_with_entities.csv → Corpus dataset.

wex_legal_terms_categorized.csv → Glossary of legal terms with definitions.

legal_examples_batch_*.csv → Per‑batch outputs with translations and metrics.

legal_examples_with_metrics.csv → Consolidated results file.

README.md → Project documentation.

** How It Works**
Load corpus + glossary.

Detect legal terms in sentences (excluding stoplist).

Retrieve examples (exact → synonyms → semantic similarity).

Translate with GPT‑4o‑mini using glossary + examples.

Score translations with BLEU, METEOR, COMET.

Save per‑batch CSVs and final consolidated CSV.

Compute corpus‑level averages for reporting.
