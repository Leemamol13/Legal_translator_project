import os
import re
import time
import pandas as pd
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from comet import download_model, load_from_checkpoint
from sentence_transformers import SentenceTransformer, util
import nltk

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

# --- Load env ---
load_dotenv(r"C:\\Users\\LENOVO\\myprojectTranslator\\.env", override=True)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- Load datasets ---
master_df = pd.read_csv("en-de_random_dataset_10k.csv")
wex_df    = pd.read_csv("wex_legal_terms_categorized.csv")

master_df["legal_category"] = master_df["legal_category"].str.strip()
wex_df["legal_category"]    = wex_df["legal_category"].str.strip()

# Filter master_df
valid_cats = wex_df["legal_category"].unique().tolist()
master_df  = master_df[master_df["legal_category"].isin(valid_cats)]
master_df  = master_df[master_df["source"].str.strip() != master_df["target"].str.strip()]
master_df  = master_df[master_df["language_pair"] == "en-de"]

# --- Load COMET ---
comet_model = None
try:
    comet_model = load_from_checkpoint(download_model("Unbabel/wmt20-comet-da"))
    print("COMET loaded")
except Exception as e:
    print(f"COMET failed: {e}")

# --- Embed wex_df (knowledge base) ---
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
wex_df["combined"] = wex_df["legal_term"].fillna("") + ". " + wex_df["definition"].fillna("")
print("Embedding wex_df knowledge base...")
wex_embeddings = embed_model.encode(
    wex_df["combined"].tolist(),
    convert_to_tensor=True,
    show_progress_bar=True
)

def safe_filename(name):
    return re.sub(r'[^\w\-]', '_', name)

# --- Similarity retrieval ---
def retrieve_by_similarity(text: str, top_k: int = 3) -> list[dict]:
    query_emb = embed_model.encode(text, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, wex_embeddings, top_k=top_k)[0]
    return [
        {
            "legal_term":     wex_df.iloc[h["corpus_id"]]["legal_term"],
            "definition":     wex_df.iloc[h["corpus_id"]]["definition"],
            "legal_category": wex_df.iloc[h["corpus_id"]]["legal_category"],
            "score":          round(h["score"], 4)
        }
        for h in hits
    ]

# --- RAG translation ---
def translate_similarity(text: str, category: str,
                         target_lang: str = "German") -> tuple[str, str]:
    retrieved = retrieve_by_similarity(text, top_k=3)

    context_block = "\n".join([
        f"- Term: '{r['legal_term']}' ({r['legal_category']}) → {r['definition']}"
        for r in retrieved
    ])

    system_prompt = (
        f"You are a professional legal translator specializing in {category}.\n"
        f"The following legal terms are semantically relevant to the text:\n"
        f"{context_block}\n"
        f"Use these terms appropriately. Return ONLY the translated text."
    )

    retrieved_terms = ", ".join([r["legal_term"] for r in retrieved])

    while True:
        try:
            resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": text}
                ],
                temperature=0.1
            )
            return resp.choices[0].message.content.strip(), retrieved_terms
        except RateLimitError as e:
            wait = 60
            m = re.search(r"try again in (\d+)m([\d.]+)s", str(e))
            if m:
                wait = int(m.group(1)) * 60 + float(m.group(2))
            print(f"[RateLimit] Waiting {wait:.0f}s...")
            time.sleep(wait)

# --- Metrics ---
def evaluate_metrics(source, candidate, reference):
    if not candidate or not reference:
        return 0.0, 0.0, 0.0

    ref_t  = word_tokenize(reference.lower())
    cand_t = word_tokenize(candidate.lower())

    bleu   = sentence_bleu([ref_t], cand_t,
                           smoothing_function=SmoothingFunction().method1)
    meteor = meteor_score([ref_t], cand_t)

    comet_score = 0.0
    if comet_model:
        try:
            out = comet_model.predict([{"src": source, "mt": candidate, "ref": reference}], batch_size=1, gpus=0)
            print("DEBUG COMET OUTPUT:", type(out), out) 

            if hasattr(out, "scores"):
                comet_score = float(out.scores[0])
            elif isinstance(out, list) and len(out) > 0:
                if isinstance(out[0], dict) and "score" in out[0]:
                    comet_score = float(out[0]["score"])
                else:
                    comet_score = float(out[0])
        except Exception as e:
            print("COMET error:", e)

    return round(bleu, 4), round(meteor, 4), round(comet_score, 4)

# --- Main loop ---
sample_sizes = [10,30,50]  
categories   = master_df["legal_category"].unique().tolist()
output_dir   = "results_csv/similarity_based"
os.makedirs(output_dir, exist_ok=True)

for category in categories[:1]:  # test only first category
    df_cat = master_df[master_df["legal_category"] == category]
    print(f"Running category: {category}, rows available: {len(df_cat)}")

    df_sample = df_cat.head(3) 
    results   = []

    for _, row in df_sample.iterrows():
        source    = str(row["source"]).strip()
        reference = str(row["target"]).strip()
        cat       = str(row["legal_category"]).strip()

        candidate, retrieved_terms = translate_similarity(source, category=cat)
        bleu, meteor, comet        = evaluate_metrics(source, candidate, reference)

        # Print row-by-row progress
        print("SRC:", source[:60])
        print("CAND:", candidate[:60])
        print("REF:", reference[:60])
        print("BLEU:", bleu, "METEOR:", meteor, "COMET:", comet)
        print("-"*50)

        results.append({
            "source":          source,
            "target":          reference,
            "language_pair":   row.get("language_pair", "en-de"),
            "legal_category":  cat,
            "scenario":        "similarity_based",
            "translation":     candidate,
            "reference":       reference,
            "bleu":            bleu,
            "meteor":          meteor,
            "comet":           comet
        })
        time.sleep(0.3)

    pd.DataFrame(results).to_csv(
        os.path.join(output_dir, f"{safe_filename(category)}_debug.csv"),
        index=False
    )
    print(f"Saved {len(df_sample)} rows → {safe_filename(category)}_debug.csv")
