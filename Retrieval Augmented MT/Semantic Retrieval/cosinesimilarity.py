import os
import re
import time
import pandas as pd
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from comet import download_model, load_from_checkpoint
from sentence_transformers import SentenceTransformer, util

# --- Load environment variables ---
load_dotenv(r"C:\\Users\\LENOVO\\myprojectTranslator\\.env", override=True)
groq_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_key)

# --- Load dataset ---
master_df = pd.read_csv("en-de_random_dataset_10k.csv")

# --- Embedding model ---
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
corpus_embeddings = embed_model.encode(master_df["source"].tolist(), convert_to_tensor=True)


def safe_filename(name: str) -> str:
    return re.sub(r'[^\w\-]', '_', name)

# --- Sample sizes ---
sample_sizes = [10, 30, 50]


model_path = download_model("Unbabel/wmt20-comet-da")
comet_model = load_from_checkpoint(model_path)

# --- Translation function ---
def translate(text, context=None, target_lang="German"):
    if context:
        context_block = "\n".join(context)
        prompt = f"Translate the following text into {target_lang}. Only return the translation, no explanations:\n\n{text}\n\nContext (for reference only, do not include in output):\n{context_block}"
    else:
        prompt = f"Translate the following text into {target_lang}. Only return the translation, no explanations:\n\n{text}"

    while True:
        try:
            resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a professional translator. Output only the translated sentence."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            candidate = resp.choices[0].message.content.strip()
            candidate = candidate.split("\n")[0] 
            return candidate
        except RateLimitError as e:
            wait_time = 60
            match = re.search(r"try again in (\d+)m(\d+\.\d+)s", str(e))
            if match:
                minutes = int(match.group(1))
                seconds = float(match.group(2))
                wait_time = minutes * 60 + seconds
            print(f"[RateLimit] Waiting {wait_time:.0f} seconds...")
            time.sleep(wait_time)

# --- Evaluation function ---
def evaluate_metrics(source, candidate, reference):
    ref_tokens = word_tokenize(reference)
    cand_tokens = word_tokenize(candidate)

    bleu = sentence_bleu([ref_tokens], cand_tokens)
    meteor = meteor_score([ref_tokens], cand_tokens)

    comet_output = comet_model.predict([{"src": source, "mt": candidate, "ref": reference}], batch_size=1)
    if isinstance(comet_output, list):
        comet_score = comet_output[0]
    elif isinstance(comet_output, dict):
        comet_score = comet_output.get("scores", [float("nan")])[0]
    else:
        comet_score = float("nan")

    ready = (bleu >= 0.25) and (meteor >= 0.35) and (comet_score >= 0.75)
    return bleu, meteor, comet_score, "PASS" if ready else "FAIL"

# --- Main loop ---
for sample_size in sample_sizes:
    results = []
    for _, row in master_df.sample(n=20, random_state=42).iterrows():
        source = row["source"]
        reference = row["target"]

        # --- Find closest sentences by cosine similarity ---
        query_embedding = embed_model.encode(source, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=min(sample_size, 10))[0]
        context_sentences = [master_df.iloc[hit["corpus_id"]]["source"] for hit in hits]

        
        candidate = translate(source, context=context_sentences)

      
        bleu, meteor, comet, status = evaluate_metrics(source, candidate, reference)

        results.append({
            "source": source,
            "target": reference,
            "translation": candidate,
            "reference": reference,
            "sample_size": sample_size,
            "bleu": bleu,
            "meteor": meteor,
            "comet": comet
        })

    out_file = f"results_cosine_{sample_size}.csv"
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"[Saved] {len(results)} translations with cosine sampling size {sample_size} → {out_file}")
