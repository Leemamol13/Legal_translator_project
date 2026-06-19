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
import numpy as np

load_dotenv(r"C:\Users\LENOVO\TranslatorProject\.env", override=True)
api_key = os.getenv("OPENAI_API_KEY")
project_id = os.getenv("OPENAI_PROJECT_ID")
client = OpenAI(api_key=api_key, project=project_id)

# --- Load datasets ---
corpus = pd.read_csv("en-de_random_dataset_10k_with_entities.csv")
glossary = pd.read_csv("wex_legal_terms_categorized.csv")

# --- Models ---
model = SentenceTransformer('all-MiniLM-L6-v2')
comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_model_path)
bleu_metric = SacreBLEU()

stoplist = {"legal", "action", "shall", "may", "hereby", "thereof", "therein", "herein",
            "see","call","test","data","take","per","basis"}
# --- translation prompt ---
def build_prompt(src, term_definitions, examples):
    defs_text = "\n".join([f"{t}: {d}" for t, d in term_definitions.items()])
    ex_text = "\n".join([f"EN: {ex['source']} | DE: {ex['target']}" for ex in examples])
    return f"""Translate the following English legal sentence into German.
Source: {src}

Definitions:
{defs_text}

Examples:
{ex_text}

Only return the German translation.
"""

# --- Translation function with definitions/examples injected ---
def translate_with_context(src, term_definitions, examples, retries=3, delay=2):
    prompt = build_prompt(src, term_definitions, examples)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0, 
                messages=[
                    {"role":"system","content":"You are a professional legal translator."},
                    {"role":"user","content":prompt}
                ]
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Translation failed (attempt {attempt+1}): {e}")
            time.sleep(delay)
    return ""


# --- WordNet synonyms ---
def get_wordnet_synonyms(term):
    syns = set()
    for syn in wordnet.synsets(term):
        for lemma in syn.lemmas():
            syns.add(lemma.name().replace("_", " "))
    return list(syns)

# --- Retrieval function ---
def get_examples_semantic(corpus, term, category, threshold=0.75):
    category_subset = corpus[corpus['legal_category'].str.contains(category, case=False, na=False)]
    if category_subset.empty:
        return pd.DataFrame()

    results = []

    # Step 1: exact word search
    exact_pattern = rf"\b{re.escape(term)}\b"
    exact_matches = category_subset[category_subset['source'].str.contains(exact_pattern, case=False, na=False)]
    results.extend(exact_matches.to_dict('records'))

    # Step 2: synonyms
    if len(results) < 5:
        terms_to_search = get_wordnet_synonyms(term)
        if terms_to_search:
            term_embeddings = model.encode(terms_to_search)
            sentence_embeddings = model.encode(category_subset['source'].tolist())
            sims = cosine_similarity(term_embeddings, sentence_embeddings).max(axis=0)
            top_indices = [i for i in sims.argsort()[::-1] if sims[i] >= threshold]
            candidates = category_subset.iloc[top_indices] if top_indices else category_subset.head(10)

            valid_synonyms = [rf"\b{re.escape(s)}\b" for s in terms_to_search]
            pattern = "|".join(valid_synonyms)
            mask = candidates['source'].str.contains(pattern, case=False, na=False)
            filtered = candidates[mask]
            results.extend(filtered.to_dict('records'))

    # Step 3: semantic fallback
    if len(results) < 5:
        sims = cosine_similarity(model.encode([term]), model.encode(category_subset['source'].tolist())).flatten()
        top_indices = sims.argsort()[::-1][:5]
        fallback = category_subset.iloc[top_indices]
        results.extend(fallback.to_dict('records'))

    # Deduplicate and cap at 5
    seen = set()
    final_results = []
    for r in results:
        if r['source'] not in seen:
            final_results.append(r)
            seen.add(r['source'])
        if len(final_results) == 5:
            break

    # --- Translation + Metrics ---
    for r in final_results:
        src, tgt = r['source'], r['target']

        # Build definitions + examples for prompt injection
        term_definitions = {term: category}
        examples_for_prompt = final_results  

        translation = translate_with_context(src, term_definitions, examples_for_prompt)

        if translation:
            cand_tokens = word_tokenize(translation)
            ref_tokens = word_tokenize(tgt)

            r['translation'] = translation
            r['BLEU'] = bleu_metric.sentence_score(translation, [tgt]).score / 100.0
            r['METEOR'] = meteor_score([ref_tokens], cand_tokens)
            r['COMET'] = comet_model.predict([{"src": src, "mt": translation, "ref": tgt}]).scores[0]
        else:
            r['translation'] = ""
            r['BLEU'] = 0.0
            r['METEOR'] = 0.0
            r['COMET'] = 0.0

    return pd.DataFrame(final_results)

# --- Definition fetch ---
def get_definition(term, glossary):
    row = glossary[glossary['legal_term'].str.lower() == term.lower()]
    if not row.empty:
        return row.iloc[0]['definition'], row.iloc[0]['legal_category']
    return "No definition found.", None

# --- Batch processing ---
batch_size = 250
all_results = []

for start in range(0, len(corpus), batch_size):
    batch = corpus.iloc[start:start+batch_size]
    logging.info(f"Processing batch {start}–{start+len(batch)-1}")

    batch_results = []

    for idx, row in batch.iterrows():
        input_sentence = row['source']
        glossary_terms = glossary['legal_term'].dropna().unique()
        legal_terms = [
            t for t in glossary_terms
            if re.search(rf"\b{re.escape(t.lower())}\b", str(input_sentence).lower())
            and t.lower() not in stoplist
        ]

        if not legal_terms:
            continue

        for term in legal_terms:
            definition, category = get_definition(term, glossary)
            if category:
                examples = get_examples_semantic(corpus, term, category)
                if not examples.empty:
                    examples['row_id'] = idx
                    examples['term'] = term
                    batch_results.extend(examples.to_dict('records'))

    if batch_results:
        all_results.extend(batch_results)
        batch_df = pd.DataFrame(batch_results)
        batch_filename = f"legal_examples_batch_{start}_{start+len(batch)-1}.csv"
        batch_df = batch_df[['source','translation','target','legal_entity','BLEU','METEOR','COMET','term']]
        batch_df.to_csv(batch_filename, index=False)
        logging.info(f"Saved batch results to {batch_filename}")
    else:
        logging.info("No legal terms found in this batch")

# --- Save all results ---
results_df = pd.DataFrame(all_results)
results_df = results_df[['source','translation','target','legal_entity','BLEU','METEOR','COMET','term']]
results_df.to_csv("legal_examples_with_metrics.csv", index=False)

# --- Corpus-level averages ---
avg_bleu = results_df['BLEU'].mean()
avg_meteor = results_df['METEOR'].mean()
avg_comet = results_df['COMET'].mean()
std_bleu = results_df['BLEU'].std()
std_meteor = results_df['METEOR'].std()
std_comet = results_df['COMET'].std()

logging.info("\n Saved results to legal_examples_with_metrics.csv with translations and metrics")
logging.info(f"Corpus-level scores: BLEU={avg_bleu:.3f}±{std_bleu:.3f}, "
             f"METEOR={avg_meteor:.3f}±{std_meteor:.3f}, "
             f"COMET={avg_comet:.3f}±{std_comet:.3f}")
