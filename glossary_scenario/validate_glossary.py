import pandas as pd
import re

# --- Load glossary ---
def load_glossary(path: str) -> dict:
    """
    Load glossary terms and categories into a dictionary.
    Keys = lowercase legal terms
    Values = category string
    """
    glossary = {}
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        term = row["legal_term"].strip().lower()
        category = row["category"].strip()
        glossary[term] = category
    return glossary

# --- Optional category corrections ---
CATEGORY_OVERRIDES = {
    "commission": "Administrative Law",
    "capital": "Corporate/Financial Law"
}

# --- Stoplist for overly generic terms ---
STOPLIST = {"member", "party", "account", "shall", "take", "listed"}

# --- Highlight terms in a sentence ---
def highlight_terms(sentence: str, glossary: dict):
    """
    Highlight glossary terms in the sentence with <Legal cat="...">term</Legal>.
    - Regex word boundaries to avoid partial matches
    - Case-insensitive matching
    - Token-based lookup for performance
    - Prevents nested replacements by sorting longest terms first
    - Post-validation to ensure no overlapping tags
    - Skips overly generic stoplist terms
    """
    highlighted = sentence
    matched = []

    # Tokenize sentence into lowercase words
    tokens = re.findall(r"\w+", sentence.lower())
    token_set = set(tokens)

    # Sort terms by length (longest first) to avoid partial overlaps
    for term, category in sorted(glossary.items(), key=lambda x: -len(x[0])):
        if term in STOPLIST:
            continue  # skip generic words
        if term in token_set:  # quick check before regex
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, highlighted, flags=re.IGNORECASE):
                # Apply category overrides if defined
                category = CATEGORY_OVERRIDES.get(term, category)
                replacement = f'<Legal cat="{category}">{term}</Legal>'
                # Replace only if not already inside a <Legal> tag
                highlighted = re.sub(
                    pattern,
                    lambda m: replacement if "<Legal" not in highlighted[m.start()-10:m.end()+10] else m.group(0),
                    highlighted,
                    flags=re.IGNORECASE
                )
                matched.append(f"{term} ({category})")

    # --- Post-validation: ensure no overlapping tags ---
    highlighted = re.sub(r'<Legal cat="[^"]+">(<Legal cat="[^"]+">.*?</Legal>)</Legal>', r'\1', highlighted)

    return highlighted, matched

# --- Process dataset ---
def process_dataset(input_csv: str, glossary_csv: str, output_csv: str):
    """
    Load sentences, highlight legal terms, and save results.
    """
    glossary = load_glossary(glossary_csv)
    df = pd.read_csv(input_csv)

    if "source" not in df.columns:
        raise ValueError("Dataset must contain a 'source' column with sentences")

    df["highlighted"], df["matched_terms"] = zip(*df["source"].apply(lambda s: highlight_terms(str(s), glossary)))

    df.to_csv(output_csv, index=False)
    print(f" Highlighted dataset saved to {output_csv}")

# --- Example run ---
if __name__ == "__main__":
    input_csv = "en-de_random_dataset_3k.csv"              
    glossary_csv = "wex_legal_terms_categorized.csv"
    output_csv = "highlighted_legal_terms.csv"

    process_dataset(input_csv, glossary_csv, output_csv)
