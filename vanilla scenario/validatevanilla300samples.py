import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from comet import download_model, load_from_checkpoint
from dotenv import load_dotenv
import os
from groq import Groq, RateLimitError
from datetime import datetime
import torch
import time

# --- Load environment variables ---
load_dotenv(r"C:\\Users\\LENOVO\\myprojectTranslator\\.env", override=True)
groq_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=groq_key)
model_name = "qwen/qwen3-32b"

# --- COMET setup ---
model_path = download_model("Unbabel/wmt20-comet-da")
comet_model = load_from_checkpoint(model_path)

# --- Evaluation function (robust COMET handling) ---
def evaluate(reference, candidate, source):
    ref_tokens = word_tokenize(reference) if reference and reference.strip() else []
    cand_tokens = word_tokenize(candidate) if candidate and candidate.strip() else []

    bleu = sentence_bleu([ref_tokens], cand_tokens) if ref_tokens and cand_tokens else 0.0
    meteor = meteor_score([ref_tokens], cand_tokens) if ref_tokens and cand_tokens else 0.0

    if not (source and source.strip() and candidate and candidate.strip() and reference and reference.strip()):
        return bleu, meteor, float("nan")

    try:
        comet_input = {"src": source.strip(), "mt": candidate.strip(), "ref": reference.strip()}
        comet_output = comet_model.predict([comet_input], batch_size=4)

        # Handle multiple return formats
        if isinstance(comet_output, list):
            comet_score = comet_output[0]
        elif isinstance(comet_output, dict):
            if "scores" in comet_output:
                comet_score = comet_output["scores"][0]
            elif "system_score" in comet_output:
                comet_score = comet_output["system_score"]
            else:
                comet_score = float("nan")
        else:
            # Some COMET versions return Prediction objects
            comet_score = getattr(comet_output, "system_score", float("nan"))

        # Ensure numeric
        if comet_score is None:
            comet_score = float("nan")

    except Exception as e:
        print("COMET failed:", e)
        comet_score = float("nan")

    return bleu, meteor, comet_score

# --- Prompt builder ---
def build_prompt(text, target_lang, scenario):
    if scenario == "direct":
        return f"Translate this text to {target_lang}:\n{text}"
    elif scenario == "domain":
        return f"Translate this legal text to {target_lang}, preserving legal terminology:\n{text}"
    elif scenario == "cot":
        return f"Step by step, explain how to translate this text to {target_lang}, then give the final translation:\n{text}"
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

# --- Translation function ---
def translate(text, target_lang="German", scenario="direct"):
    prompt = build_prompt(text, target_lang, scenario)
    while True:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a professional translator"},
                    {"role": "user", "content": prompt}
                ]
            )
            return resp.choices[0].message.content.strip()
        except RateLimitError:
            print("⚠️ Rate limit hit, waiting 120s...")
            time.sleep(120)

# --- Dataset ---
df = pd.read_csv("en-de_random_dataset_3k.csv")
df_sample = df.sample(n=300, random_state=42)

batch_size = 50
results = []
scenarios = ["direct", "domain", "cot"]

for start in range(0, len(df_sample), batch_size):
    end = start + batch_size
    batch = df_sample.iloc[start:end]

    for i, row in batch.iterrows():
        source = row["source"]
        reference = row["target"]

        row_result = {"source": source, "reference": reference}

        for scenario in scenarios:
            candidate = translate(source, scenario=scenario).strip()
            if scenario == "cot":
                candidate = candidate.split("\n")[-1].strip()

            bleu, meteor, comet = evaluate(reference, candidate, source)

            row_result[f"{scenario}_translation"] = candidate
            row_result[f"{scenario}_bleu"] = bleu
            row_result[f"{scenario}_meteor"] = meteor
            row_result[f"{scenario}_comet"] = comet

            print(f"[Batch {start//batch_size+1}] Row {i} ({scenario}): "
                  f"BLEU={bleu:.4f}, METEOR={meteor:.4f}, COMET={comet:.4f}")

        results.append(row_result)

    file_name = f"results_groq_part_{start//batch_size+1}.csv"
    pd.DataFrame(results).to_csv(file_name, index=False)
    print(f"Saved batch {start//batch_size+1} ({start}–{end}) → {file_name}")

    torch.cuda.empty_cache()
    del batch

print("✅ Groq evaluation complete. Results saved in Groq CSVs.")

# --- Summary ---
df_results = pd.DataFrame(results)
summary = {f"{s}_bleu_avg": df_results[f"{s}_bleu"].mean() for s in scenarios}
summary.update({f"{s}_meteor_avg": df_results[f"{s}_meteor"].mean() for s in scenarios})
summary.update({f"{s}_comet_avg": df_results[f"{s}_comet"].mean() for s in scenarios})

print("\n=== Overall Evaluation Summary (300 rows) ===")
for k, v in summary.items():
    print(f"{k}: {v:.4f}")
