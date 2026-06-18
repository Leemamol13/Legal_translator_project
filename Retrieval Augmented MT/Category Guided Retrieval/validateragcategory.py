import pandas as pd
import os
import re
from dotenv import load_dotenv
from groq import Groq
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from comet import download_model, load_from_checkpoint
from tqdm import tqdm
import nltk


nltk.download("punkt", quiet=True)

# --- Load environment variables ---
load_dotenv(r"C:\\Users\\LENOVO\\myprojectTranslator\\.env", override=True)
groq_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_key)

# --- Load datasets ---
master_df = pd.read_csv("en-de_random_dataset_10k.csv")  
wex_df = pd.read_csv("wex_legal_terms_categorized.csv") 
print("Columns:", master_df.columns.tolist())


def safe_filename(name: str) -> str:
    return re.sub(r'[^\w\-]', '_', name)

valid_categories = wex_df["legal_category"].unique().tolist()
master_df = master_df[master_df["legal_category"].isin(valid_categories)]

categories = master_df["legal_category"].unique().tolist()
print("Detected categories:", categories)

sample_sizes = [10, 30, 50]

comet_model = None
try:
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    comet_model = load_from_checkpoint(model_path)
    print("Loaded COMET model: wmt22-cometkiwi-da")
except Exception as e:
    print("COMET failed to load:", e)

# --- Translation function ---
def translate(text, target_lang="German"):
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role":"system","content":"You are a professional translator"},
                {"role":"user","content":f"Translate to {target_lang}: {text}"}
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f" Translation failed: {e}")
        return ""

# --- Evaluation function ---
def evaluate_metrics(source, candidate, reference):
    ref_tokens = word_tokenize(reference) if reference else []
    cand_tokens = word_tokenize(candidate) if candidate else []

    bleu = sentence_bleu([ref_tokens], cand_tokens,
                         smoothing_function=SmoothingFunction().method1) if ref_tokens and cand_tokens else 0.0
    meteor = meteor_score([ref_tokens], cand_tokens) if ref_tokens and cand_tokens else 0.0

    comet_score = None
    if comet_model:
        try:
            output = comet_model.predict([{"src": source, "mt": candidate, "ref": reference}], batch_size=1)
            if isinstance(output, list) and len(output) > 0:
                comet_score = float(output[0])
        except Exception as e:
            print("COMET failed:", e)

    return bleu, meteor, comet_score


# --- Save results as CSV files ---
output_dir = "results_csv"
os.makedirs(output_dir, exist_ok=True)

for category in categories:
    df_category = master_df[master_df["legal_category"] == category]

    for sample_size in sample_sizes:
        actual_size = min(sample_size, len(df_category))
        safe_cat = safe_filename(category)
        filename = os.path.join(output_dir, f"{safe_cat}_{sample_size}.csv")

        if actual_size == 0:
            pd.DataFrame([{
                "source": None,
                "target": None,
                "language_pair": None,
                "legal_category": category,
                "translation": None,
                "reference": None,
                "bleu": None,
                "meteor": None,
                "comet": None,
                "note": "No data available for this category"
            }]).to_csv(filename, index=False)
            print(f"No rows found for {category}, created empty CSV → {filename}")
            continue

        df_sample = df_category.sample(n=actual_size, random_state=42)
        results = []

        for _, row in tqdm(df_sample.iterrows(), total=actual_size, desc=f"{category} {sample_size}"):
            source = row["source"]
            reference = row["target"]
            candidate = translate(source)

            bleu, meteor, comet = evaluate_metrics(source, candidate, reference)

            results.append({
                "source": source,
                "target": reference,
                "language_pair": row["language_pair"],
                "legal_category": category,
                "translation": candidate,
                "reference": reference,
                "bleu": bleu,
                "meteor": meteor,
                "comet": comet if comet is not None else None
            })

        pd.DataFrame(results).to_csv(filename, index=False)
        print(f"Saved {actual_size} translations for category {category} → {filename}")
