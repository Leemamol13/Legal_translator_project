# -----------------------------
# all imports stay at top
# -----------------------------
import pandas as pd
import os
import re
import traceback
from dotenv import load_dotenv
from groq import Groq
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from comet import download_model, load_from_checkpoint
from tqdm import tqdm
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
from groq import RateLimitError, InternalServerError

# -----------------------------
# Setup — stays outside
# -----------------------------
nltk.download("punkt", quiet=True)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
load_dotenv(r"C:\Users\LENOVO\myprojectTranslator\.env", override=True)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# Load datasets — stays outside
# -----------------------------
master_df = pd.read_csv("en-de_random_dataset_10k.csv")
wex_df    = pd.read_csv("wex_legal_terms_categorized.csv")

# sample_sizes removed ✅
master_df["legal_category"] = master_df["legal_category"].str.strip()
wex_df["legal_category"]    = wex_df["legal_category"].str.strip()
valid_categories = wex_df["legal_category"].dropna().unique()
master_df = master_df[master_df["legal_category"].isin(valid_categories)]

# -----------------------------
# Functions — stay outside
# -----------------------------
def retrieve_category_examples(source, category, k=3):
    data = master_df[master_df["legal_category"] == category]
    data = data[data["source"] != source]
    if data.empty:
        return []
    sentences           = data["source"].tolist()
    query_embedding     = embedding_model.encode([source])
    sentence_embeddings = embedding_model.encode(sentences)
    scores              = cosine_similarity(query_embedding, sentence_embeddings)[0]
    best_ids = [i for i in scores.argsort()[::-1] if scores[i] >= 0.45][:k]
    return [{"source": data.iloc[i]["source"], "target": data.iloc[i]["target"]} for i in best_ids]

def translate_rag(source, examples):
    if examples:
        example_text = "\n".join([
            f"English:\n{x['source']}\n\nGerman:\n{x['target']}"
            for x in examples
        ])
    else:
        example_text = "No examples available"

    prompt = f"""
You are a professional legal translator.
Translate the English legal sentence into accurate German.
Use the retrieved examples only as guidance.

Examples:
{example_text}

Sentence:
{source}

Return only German translation.
"""
    wait = 30
    while True:
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You translate legal documents."},
                    {"role": "user",   "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()

        except RateLimitError:
            print(f"Rate limit hit, waiting {wait}s...")
            time.sleep(wait)
            wait = min(wait * 2, 300)  # exponential backoff, max 5 min

        except InternalServerError:
            print(f"Groq 503 overloaded, waiting {wait}s...")
            time.sleep(wait)
            wait = min(wait * 2, 300)  # exponential backoff, max 5 min


def evaluate_metrics(source, candidate, reference):
    ref_tokens  = word_tokenize(reference)
    cand_tokens = word_tokenize(candidate)
    bleu   = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=SmoothingFunction().method1)
    meteor = meteor_score([ref_tokens], cand_tokens)
    comet_score = None
    if comet_model:
        try:
            result      = comet_model.predict([{"src": source, "mt": candidate, "ref": reference}])
            comet_score = float(result.scores[0])
        except Exception as e:
            print("COMET error:", e)
    return bleu, meteor, comet_score

# -------------------------------------------------------
# EVERYTHING BELOW goes inside if __name__ == "__main__"
# -------------------------------------------------------
if __name__ == "__main__":

    # COMET loads here
    comet_model = None
    try:
        model_path  = download_model("Unbabel/wmt20-comet-da")
        comet_model = load_from_checkpoint(model_path)
        print("COMET loaded")
    except Exception as e:
        print("COMET loading failed:", e)

    print("Available rows:", len(master_df))
    
    categories = master_df["legal_category"].unique().tolist()  # ✅ defined here
    print("Categories:", categories)

    try:
        for category in categories:                             # ✅ correct indent
            print(f"\nProcessing category: {category}")

            df_cat    = master_df[master_df["legal_category"] == category]
            sample_df = df_cat.sample(
                n=min(20, len(df_cat)),
                random_state=42
            ).reset_index(drop=True)

            results = []                                        # ✅ reset per category

            for _, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
                source    = row["source"]
                reference = row["target"]

                examples    = retrieve_category_examples(source, category, k=3)
                translation = translate_rag(source, examples)
                bleu, meteor, comet = evaluate_metrics(source, translation, reference)

                results.append({
                    "source":             source,
                    "reference":          reference,
                    "category":           category,
                    "retrieved_examples": len(examples),
                    "translation":        translation,
                    "BLEU":               bleu,
                    "METEOR":             meteor,
                    "COMET":              comet
                })

            # save one file per category ✅
            safe_cat = category.replace(" ", "_").replace("/", "_").replace("&", "and")
            output   = pd.DataFrame(results)
            output.to_csv(f"rag_mt_{safe_cat}.csv", index=False)
            print(f"Saved: rag_mt_{safe_cat}.csv")
            print(output[["BLEU", "METEOR", "COMET"]].mean())

        # summary across all categories ✅
        all_files  = [f for f in os.listdir() if f.startswith("rag_mt_") and f.endswith(".csv")]
        all_results = pd.concat([pd.read_csv(f) for f in all_files])
        summary     = all_results.groupby("category")[["BLEU", "METEOR", "COMET"]].mean().round(4)
        print("\n=== Category-wise Performance ===")
        print(summary)
        summary.to_csv("rag_mt_category_summary.csv")
        print("Summary saved: rag_mt_category_summary.csv")

    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()