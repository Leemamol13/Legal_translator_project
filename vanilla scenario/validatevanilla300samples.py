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
from groq import Groq
from datetime import datetime
import torch

# --- Load environment variables ---
load_dotenv(r"C:\Users\LENOVO\myprojectTranslator\.env", override=True)
groq_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=groq_key)
model_name = "qwen/qwen3-32b"

# --- COMET setup
try:
    model_path = download_model("Unbabel/wmt20-comet-da")  
except Exception as e:
    raise RuntimeError(f"Failed to load COMET model: {e}")

comet_model = load_from_checkpoint(model_path)


# --- Evaluation function ---
def evaluate(reference, candidate, source):
    ref_tokens = word_tokenize(reference)
    cand_tokens = word_tokenize(candidate)

    bleu = sentence_bleu([ref_tokens], cand_tokens)
    meteor = meteor_score([ref_tokens], cand_tokens)

    try:
        comet_score = comet_model.predict(
            [{"src": source, "mt": candidate, "ref": reference}],
            batch_size=1,   
            gpus=0          
        ).system_score
    except RuntimeError as e:
        print("COMET failed due to memory:", e)
        comet_score = float("nan")

    return bleu, meteor, comet_score

# --- Translation function ---
def translate(text, target_lang="German"):
    prompt = f"Translate this text to {target_lang}:\n{text}"
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role":"system","content":"You are a professional translator"},
            {"role":"user","content":prompt}
        ]
    )
    return resp.choices[0].message.content.strip()

# --- Load dataset in chunks and sample 300 rows ---
chunks = pd.read_csv("en-de_random_dataset_3k.csv", chunksize=1000)
df = pd.concat(chunks, ignore_index=True)
df_sample = df.sample(n=300, random_state=42)

batch_size = 50  
results = []

for start in range(0, len(df_sample), batch_size):
    end = start + batch_size
    batch = df_sample.iloc[start:end]

    for i, row in batch.iterrows():
        source = row["source"]
        reference = row["target"]

        out_groq = translate(source)
        bleu_g, meteor_g, comet_g = evaluate(reference, out_groq, source)

        results.append({
            "source": source,
            "reference": reference,
            "groq_bleu": bleu_g,
            "groq_meteor": meteor_g,
            "groq_comet": comet_g
        })

        print(f"[Batch {start//batch_size+1}] Row {i}: BLEU={bleu_g:.4f}, METEOR={meteor_g:.4f}, COMET={comet_g:.4f}")

    file_name = f"results_groq_part_{start//batch_size+1}.csv"
    if os.path.exists(file_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"results_groq_part_{start//batch_size+1}_{timestamp}.csv"

    pd.DataFrame(results).to_csv(file_name, index=False)
    print(f"Saved batch {start//batch_size+1} ({start}–{end}) → {file_name}")

    torch.cuda.empty_cache()
    del batch

print(" Groq evaluation complete. Results saved in Groq CSVs.")
