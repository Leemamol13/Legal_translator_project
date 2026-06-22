import pandas as pd
import re
import os
import time
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from comet import download_model, load_from_checkpoint

# --- Load environment variables ---
load_dotenv(r"C:\\Users\\LENOVO\\myprojectTranslator\\.env", override=True)
groq_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_key)

# --- Load COMET model ---
try:
    model_path = download_model("Unbabel/wmt20-comet-da")
    comet_model = load_from_checkpoint(model_path)
    print("Loaded COMET model: wmt20-comet-da")
except Exception as e:
    print("COMET model failed:", e)
    comet_model = None

# --- Glossary loader ---
def load_glossary(path: str) -> dict:
    glossary = {}
    df = pd.read_csv(path)

    term_col = "legal_term"
    category_col = "legal_category" if "legal_category" in df.columns else "category"
    definition_col = "definition" if "definition" in df.columns else None

    for _, row in df.iterrows():
        term = str(row[term_col]).strip().lower()
        category = str(row[category_col]).strip()
        definition = ""
        if definition_col:
            val = row[definition_col]
            if pd.notna(val):
                definition = str(val).strip()
        glossary[term] = {"category": category, "definition": definition}
    return glossary

CATEGORY_OVERRIDES = {"commission": "Administrative Law", "capital": "Corporate/Financial Law"}
STOPLIST = {"member", "party", "account", "shall", "take", "listed"}

# --- Highlight terms with optional definition injection ---
def highlight_terms(sentence: str, glossary: dict, inject_definitions: bool = True):
    highlighted = sentence
    matched = []
    injected_definitions = []

    tokens = re.findall(r"\w+", sentence.lower())
    token_set = set(tokens)

    for term, info in sorted(glossary.items(), key=lambda x: -len(x[0])):
        if term in STOPLIST:
            continue
        if re.search(r"\b" + re.escape(term) + r"\b", sentence, flags=re.IGNORECASE):
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, highlighted, flags=re.IGNORECASE):
                category = CATEGORY_OVERRIDES.get(term, info["category"])
                replacement = f'<Legal cat="{category}">{term}</Legal>'
                highlighted = re.sub(
                    pattern,
                    lambda m: replacement if "<Legal" not in highlighted[m.start()-10:m.end()+10] else m.group(0),
                    highlighted,
                    flags=re.IGNORECASE
                )
                matched.append(f"{term} ({category})")
                if inject_definitions and info["definition"]:
                    injected_definitions.append(f"{term}: {info['definition']}")

    highlighted = re.sub(r'<Legal cat="[^"]+">(<Legal cat="[^"]+">.*?</Legal>)</Legal>', r'\1', highlighted)
    return highlighted, matched, injected_definitions

# --- Scenario 1: Context-Aware Glossary Translation ---
def translate_glossary_context(text, matched_terms, target_lang="German"):
    glossary_block = "\n".join([f"- {term}" for term in matched_terms])
    prompt = f"""
You are a professional legal translator.

Relevant legal terminology:
{glossary_block}

Translate the following text into {target_lang}.
Use the legal terminology context when selecting translations.
Return ONLY the translated text.

Text:
{text}
"""
    while True:
        try:
            resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "You are an expert legal translator."},
                    {"role": "user", "content": prompt}
                ]
            )
            return resp.choices[0].message.content.strip()
        except RateLimitError:
            print("Rate limit hit, waiting 120s...")
            time.sleep(120)

# --- Scenario 2: Definition Injection Translation ---
def translate_definition_injection(text, definitions, target_lang="German"):
    definition_block = "\n".join([f"- {d}" for d in definitions])
    prompt = f"""
You are a professional legal translator.

Relevant legal definitions:
{definition_block}

Translate the following text into {target_lang}.
Use the definitions to choose the most appropriate legal terminology.
Return ONLY the translated text.

Text:
{text}
"""
    while True:
        try:
            resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "You are an expert legal translator."},
                    {"role": "user", "content": prompt}
                ]
            )
            return resp.choices[0].message.content.strip()
        except RateLimitError:
            print("Rate limit hit, waiting 120s...")
            time.sleep(120)

# --- Evaluation function ---
def evaluate_metrics(source, candidate, reference):
    ref_tokens = word_tokenize(reference.lower()) if reference else []
    cand_tokens = word_tokenize(candidate.lower()) if candidate else []

    smoothie = SmoothingFunction().method1
    bleu = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie) if ref_tokens and cand_tokens else 0.0
    meteor = meteor_score([ref_tokens], cand_tokens) if ref_tokens and cand_tokens else 0.0

    comet_score = None
    if comet_model:
        try:
            comet_score = comet_model.predict(
                [{"src": source, "mt": candidate, "ref": reference}],
                batch_size=1
            )[0]
        except Exception as e:
            print("COMET failed:", e)

    return bleu, meteor, comet_score

# --- Process dataset ---
def process_dataset(input_csv: str, glossary_csv: str, output_csv: str, sample_size: int = 10):
    glossary = load_glossary(glossary_csv)
    df = pd.read_csv(input_csv)

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Running on a random sample of {sample_size} rows (out of {len(pd.read_csv(input_csv))})")

    results = []
    for _, row in df.iterrows():
        source = str(row["source"])
        reference = str(row["target"]) if "target" in row else ""

        # Scenario 1: Context-Aware Glossary
        highlighted_ctx, matched_ctx, defs_ctx = highlight_terms(source, glossary, inject_definitions=False)
        translation_glossary = translate_glossary_context(source, matched_ctx)
        bleu_ctx, meteor_ctx, comet_ctx = evaluate_metrics(source, translation_glossary, reference)

        # Scenario 2: Definition Injection
        highlighted_def, matched_def, defs_def = highlight_terms(source, glossary, inject_definitions=True)
        translation_definition = translate_definition_injection(source, defs_def)
        bleu_def, meteor_def, comet_def = evaluate_metrics(source, translation_definition, reference)

        results.append({
            "source": source,
            "reference": reference,
            # Context-Aware Glossary
            "highlighted_context": highlighted_ctx,
            "matched_terms_context": matched_ctx,
            "translation_glossary": translation_glossary,
            "bleu_context": bleu_ctx,
            "meteor_context": meteor_ctx,
            "comet_context": comet_ctx,
            # Definition Injection
            "highlighted_definitions": highlighted_def,
            "matched_terms_definitions": matched_def,
            "definitions_injected": defs_def,
            "translation_definition": translation_definition,
            "bleu_definitions": bleu_def,
            "meteor_definitions": meteor_def,
            "comet_definitions": comet_def
        })

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Results saved with both scenarios → {output_csv}")

if __name__ == "__main__":
    input_csv = "en-de_random_dataset_3k.csv"
    glossary_csv = "wex_legal_terms_categorized.csv"
    output_csv = "sample_results.csv"
    process_dataset(input_csv, glossary_csv, output_csv, sample_size=10)
