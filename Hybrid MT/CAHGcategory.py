import pandas as pd
import re
import os
import time
import traceback
import nltk
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from comet import download_model, load_from_checkpoint
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

load_dotenv(r"C:\Users\LENOVO\myprojectTranslator\.env", override=True)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

STOPLIST = {"member", "party", "account", "shall", "take", "listed"}

# -----------------------------
# Load glossary
# -----------------------------
def load_glossary(path):
    df = pd.read_csv(path)
    glossary = {}
    for _, row in df.iterrows():
        term     = str(row["legal_term"]).strip().lower()
        category = str(row["legal_category"]).strip()
        definition = str(row["definition"]).strip() if pd.notna(row.get("definition")) else ""
        glossary[term] = {"category": category, "definition": definition}
    return glossary

# -----------------------------
# Detect terms in sentence
# -----------------------------
def detect_terms(sentence, glossary):
    terms = {}
    for term, info in sorted(glossary.items(), key=lambda x: -len(x[0])):
        if term in STOPLIST:
            continue
        if re.search(r"\b" + re.escape(term) + r"\b", sentence, flags=re.IGNORECASE):
            terms[term] = info
    return terms

# -----------------------------
# Category-filtered retrieval
# -----------------------------
def retrieve_category_examples(source, category, corpus_df, k=3):
    data = corpus_df[corpus_df["legal_category"] == category]
    data = data[data["source"] != source]
    if data.empty:
        return []
    sentences           = data["source"].tolist()
    query_emb           = embedding_model.encode([source])
    sentence_embs       = embedding_model.encode(sentences)
    scores              = cosine_similarity(query_emb, sentence_embs)[0]
    best_ids            = [i for i in scores.argsort()[::-1] if scores[i] >= 0.45][:k]
    return [{"source": data.iloc[i]["source"], "target": data.iloc[i]["target"]} for i in best_ids]

# -----------------------------
# CAHG Translation
# -----------------------------
def translate_cahg(source, category, definitions, examples, target_lang="German"):
    def_block = "\n".join([f"- {t}: {d}" for t, d in definitions.items()]) or "None"
    ex_block  = "\n".join([
        f"English: {e['source']}\nGerman: {e['target']}"
        for e in examples
    ]) or "No examples available"

    prompt = f"""
You are a professional legal translator specializing in {category}.

You are translating a sentence from the legal domain: {category}.

Relevant legal definitions:
{def_block}

Example translations from {category}:
{ex_block}

Translate the following English sentence into {target_lang}.
Use the category context, definitions, and examples as guidance.
Return ONLY the translated text.

Sentence:
{source}
"""
    wait = 30
    while True:
        try:
            resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                messages=[
                    {"role": "system", "content": f"You are an expert legal translator in {category}."},
                    {"role": "user",   "content": prompt}
                ]
            )
            return resp.choices[0].message.content.strip()
        except RateLimitError:
            print(f"Rate limit, waiting {wait}s...")
            time.sleep(wait)
            wait = min(wait * 2, 300)
        except Exception as e:
            print(f"Groq error, waiting {wait}s... {e}")
            time.sleep(wait)
            wait = min(wait * 2, 300)

# -----------------------------
# Metrics
# -----------------------------
def evaluate_metrics(source, candidate, reference, comet_model):
    ref_tokens  = word_tokenize(reference.lower()) if reference else []
    cand_tokens = word_tokenize(candidate.lower()) if candidate else []
    bleu   = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=SmoothingFunction().method1) if ref_tokens and cand_tokens else 0.0
    meteor = meteor_score([ref_tokens], cand_tokens) if ref_tokens and cand_tokens else 0.0
    comet_score = None
    if comet_model:
        try:
            result      = comet_model.predict([{"src": source, "mt": candidate, "ref": reference}])
            comet_score = float(result.scores[0])
        except Exception as e:
            print("COMET error:", e)
    return bleu, meteor, comet_score

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    comet_model = None
    try:
        comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        print("COMET loaded")
    except Exception as e:
        print("COMET failed:", e)

    glossary  = load_glossary("wex_legal_terms_categorized.csv")
    corpus_df = pd.read_csv("en-de_random_dataset_10k.csv")
    corpus_df["legal_category"] = corpus_df["legal_category"].str.strip()
    categories = corpus_df["legal_category"].dropna().unique().tolist()

    try:
        for category in categories:
            print(f"\nCAHG — Processing: {category}")
            df_cat    = corpus_df[corpus_df["legal_category"] == category]
            sample_df = df_cat.sample(n=min(20, len(df_cat)), random_state=42).reset_index(drop=True)
            results   = []

            for _, row in sample_df.iterrows():
                source    = str(row["source"])
                reference = str(row["target"])

                # Step 1: detect terms
                detected = detect_terms(source, glossary)

                # Step 2: get definitions
                definitions = {t: info["definition"] for t, info in detected.items() if info["definition"]}

                # Step 3: retrieve category-filtered examples
                examples = retrieve_category_examples(source, category, corpus_df, k=3)

                # Step 4: translate with category context
                translation = translate_cahg(source, category, definitions, examples)

                # Step 5: evaluate
                bleu, meteor, comet = evaluate_metrics(source, translation, reference, comet_model)

                results.append({
                    "source":            source,
                    "reference":         reference,
                    "category":          category,
                    "detected_terms":    list(detected.keys()),
                    "definitions_used":  definitions,
                    "examples_retrieved": len(examples),
                    "translation":       translation,
                    "BLEU":              bleu,
                    "METEOR":            meteor,
                    "COMET":             comet
                })

            safe_cat = category.replace(" ", "_").replace("/", "_").replace("&", "and")
            output   = pd.DataFrame(results)
            output.to_csv(f"cahg_{safe_cat}.csv", index=False)
            print(f"Saved: cahg_{safe_cat}.csv")
            print(output[["BLEU", "METEOR", "COMET"]].mean())

    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()