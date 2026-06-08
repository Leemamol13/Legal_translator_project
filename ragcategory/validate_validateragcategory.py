import pandas as pd
import os
import re
from dotenv import load_dotenv
from groq import Groq
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from comet import download_model, load_from_checkpoint

# --- Load environment variables ---
load_dotenv(r"C:\\Users\\LENOVO\\myprojectTranslator\\.env", override=True)
groq_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_key)

# --- Load datasets ---
master_df = pd.read_csv("en-de_random_dataset_10k.csv")  
wex_df = pd.read_csv("wex_legal_terms_categorized.csv") 
print(master_df.columns.tolist())


# --- Safe filename helper ---
def safe_filename(name: str) -> str:
   
    return re.sub(r'[^\w\-]', '_', name)


valid_categories = wex_df["category"].unique().tolist()
master_df = master_df[master_df["legal_category"].isin(valid_categories)]

# --- Categories to process ---
categories = [
    "Legal Theory & Latin Terms",
    "Property & Real Estate Law",
    "Intellectual Property (IP)",
    "Family Law & Estates/Trusts",
    "Bankruptcy Law",
    "Civil Procedure & Litigation",
    "Contract Law"
]

# --- Sample sizes ---
sample_sizes = [10, 50]

# --- Load COMET model once ---
model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)

# --- Translation function ---
def translate(text, target_lang="German"):
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role":"system","content":"You are a professional translator"},
            {"role":"user","content":f"Translate to {target_lang}: {text}"}
        ]
    )
    return resp.choices[0].message.content.strip()

# --- Evaluation function---
def evaluate_metrics(source, candidate, reference):
    
    ref_tokens = word_tokenize(reference)
    cand_tokens = word_tokenize(candidate)

    bleu = sentence_bleu([ref_tokens], cand_tokens)
    meteor = meteor_score([ref_tokens], cand_tokens)
    comet_score = comet_model.predict([{"src": source, "mt": candidate, "ref": reference}])[0]
    return bleu, meteor, comet_score

# --- Loop through categories and sample sizes ---
for category in categories:
    df_category = master_df[master_df["legal_category"] == category]

    for sample_size in sample_sizes:
        # If category has fewer rows than requested, use all available rows
        actual_size = min(sample_size, len(df_category))
        safe_cat = safe_filename(category)

        if actual_size == 0:
            # Save an empty file with a note so category is still represented
            out_file = f"results_{safe_cat}_{sample_size}.csv"
            pd.DataFrame([{
                "source": None,
                "target": None,
                "language_pair": None,
                "Legal_Category": category,
                "translation": None,
                "reference": None,
                "bleu": None,
                "meteor": None,
                "comet": None,
                "note": "No data available for this category"
            }]).to_csv(out_file, index=False)
            print(f" No rows found for {category}, created empty file → {out_file}")
            continue

        df_sample = df_category.sample(n=actual_size, random_state=42)
        results = []

        for _, row in df_sample.iterrows():
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
                "comet": comet
            })

        out_file = f"results_{safe_cat}_{sample_size}.csv"
        pd.DataFrame(results).to_csv(out_file, index=False)
        print(f" Saved {actual_size} translations for category {category} → {out_file}")
