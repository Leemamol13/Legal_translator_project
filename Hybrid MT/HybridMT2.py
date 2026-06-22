import pandas as pd
import re
import logging
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sacrebleu.metrics import BLEU as SacreBLEU
from nltk.translate.meteor_score import meteor_score
from comet import download_model, load_from_checkpoint
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from openai import OpenAI
import os
from dotenv import load_dotenv

# -----------------------------
# Setup
# -----------------------------
load_dotenv(r"C:\Users\LENOVO\TranslatorProject\.env", override=True)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    project=os.getenv("OPENAI_PROJECT_ID")
)

# -----------------------------
# Load data
# -----------------------------
corpus = pd.read_csv("en-de_random_dataset_10k_with_entities.csv")
glossary = pd.read_csv("wex_legal_terms_categorized.csv")

corpus = corpus.sample(n=100, random_state=42).reset_index(drop=True)
print(f"Running Hybrid MT on {len(corpus)} samples")

# -----------------------------
# Models
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

comet_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_path)

bleu_metric = SacreBLEU()

STOPLIST = {
    "legal", "action", "shall", "may", "hereby", "thereof",
    "therein", "herein", "see", "call", "test", "data",
    "take", "per", "basis"
}

# -----------------------------
# Prompt
# -----------------------------
def build_prompt(source, definitions, examples):
    defs = "\n".join([f"{k}: {v}" for k, v in definitions.items()])
    ex = "\n".join([f"EN: {x['source']}\nDE: {x['target']}" for x in examples])

    return f"""
You are a professional legal translator.
Translate English legal text into German.
Use the provided legal definitions and examples.

Definitions:
{defs}

Examples:
{ex}

Source:
{source}

Return only German translation.
"""

# -----------------------------
# Translation
# -----------------------------
def translate_hybrid(source, definitions, examples):
    prompt = build_prompt(source, definitions, examples)
    for i in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": "You translate legal documents."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(e)
            time.sleep(2)
    return ""

# -----------------------------
# Wordnet synonyms
# -----------------------------
def get_synonyms(term):
    result = set()
    for syn in wordnet.synsets(term):
        for lemma in syn.lemmas():
            result.add(lemma.name().replace("_", " "))
    return list(result)

# -----------------------------
# Retrieve examples
# -----------------------------
def retrieve_examples(corpus, term, category, top_k=5):
    subset = corpus[corpus["legal_category"].str.contains(category, case=False, na=False)]
    if subset.empty:
        return []

    results = []

    # 1. Exact term search
    pattern = rf"\b{re.escape(term)}\b"
    exact = subset[subset["source"].str.contains(pattern, case=False, na=False)]
    results.extend(exact.to_dict("records"))

    # 2. Similar meaning search
    if len(results) < top_k:
        synonyms = get_synonyms(term)
        search_terms = [term] + synonyms
        term_embeddings = embedding_model.encode(search_terms)
        sentence_embeddings = embedding_model.encode(subset["source"].tolist())
        similarity = cosine_similarity(term_embeddings, sentence_embeddings)
        max_scores = similarity.max(axis=0)
        best_idx = max_scores.argsort()[::-1]

        for idx in best_idx:
            if max_scores[idx] > 0.65:
                results.append(subset.iloc[idx].to_dict())
            if len(results) == top_k:
                break

    # Remove duplicates
    final, seen = [], set()
    for r in results:
        if r["source"] not in seen:
            final.append(r)
            seen.add(r["source"])
        if len(final) == top_k:
            break

    return final

# -----------------------------
# Glossary lookup
# -----------------------------
def get_definition(term):
    row = glossary[glossary["legal_term"].str.lower() == term.lower()]
    if not row.empty:
        return row.iloc[0]["definition"], row.iloc[0]["legal_category"]
    return None, None

# -----------------------------
# Evaluation
# -----------------------------
def evaluate(source, translation, reference):
    bleu = bleu_metric.sentence_score(translation, [reference]).score / 100
    meteor = meteor_score([word_tokenize(reference)], word_tokenize(translation))
    comet = comet_model.predict([{"src": source, "mt": translation, "ref": reference}]).scores[0]
    return bleu, meteor, comet

# -----------------------------
# Processing
# -----------------------------
results = []

for idx, row in corpus.iterrows():
    source = row["source"]
    reference = row["target"]

    terms = []
    for t in glossary["legal_term"].dropna():
        if re.search(rf"\b{re.escape(t.lower())}\b", source.lower()) and t.lower() not in STOPLIST:
            terms.append(t)

    if not terms:
        continue

    definitions = {}
    all_examples = []

    for term in terms:
        definition, category = get_definition(term)
        if definition:
            definitions[term] = definition
            examples = retrieve_examples(corpus, term, category)
            all_examples.extend(examples)

            translation = translate_hybrid(source, definitions, all_examples[:5])
            bleu, meteor, comet = evaluate(source, translation, reference)

            results.append({
                "source": source,
                "reference": reference,
                "term": term,
                "definition_used": definition,
                "retrieved_examples": len(examples),
                "translation": translation,
                "BLEU": bleu,
                "METEOR": meteor,
                "COMET": comet
            })

# -----------------------------
# Save
# -----------------------------
df = pd.DataFrame(results)
df.to_csv("hybrid_definition_example_MT_results.csv", index=False)
print("Hybrid MT evaluation completed")
