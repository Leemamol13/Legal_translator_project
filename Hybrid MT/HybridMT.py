import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from comet import download_model, load_from_checkpoint
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

# --- Load datasets ---
corpus = pd.read_csv("en-de_random_dataset_10k_with_entities.csv")   # source, target, language_pair, legal_category, legal_entity
glossary = pd.read_csv("wex_legal_terms_categorized.csv")  # legal_term, legal_category, definition

model = SentenceTransformer('all-MiniLM-L6-v2')
comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_model_path)

stoplist = {"legal", "action", "shall", "may", "hereby", "thereof", "therein", "herein",
            "see","call","test","data","take","per","basis"}


def get_wordnet_synonyms(term):
    syns = set()
    for syn in wordnet.synsets(term):
        for lemma in syn.lemmas():
            syns.add(lemma.name().replace("_", " "))
    return list(syns)

# --- Retrieval function  ---
def get_examples_semantic(corpus, term, category, threshold=0.75):
    category_subset = corpus[corpus['legal_category'].str.contains(category, case=False, na=False)]
    if category_subset.empty:
        return pd.DataFrame()

    results = []

    # Step 1: exact word search
    exact_pattern = rf"\b{re.escape(term)}\b"
    exact_matches = category_subset[category_subset['source'].str.contains(exact_pattern, case=False, na=False)]
    results.extend(exact_matches.to_dict('records'))

    # Step 2: WordNet synonyms if fewer than 5
    if len(results) < 5:
        terms_to_search = get_wordnet_synonyms(term)
        if terms_to_search:
            term_embeddings = model.encode(terms_to_search)
            sentence_embeddings = model.encode(category_subset['source'].tolist())
            sims = cosine_similarity(term_embeddings, sentence_embeddings).max(axis=0)
            top_indices = [i for i in sims.argsort()[::-1] if sims[i] >= threshold]
            candidates = category_subset.iloc[top_indices] if top_indices else category_subset.head(10)

            # whole-word synonym filter
            valid_synonyms = [rf"\b{re.escape(s)}\b" for s in terms_to_search]
            pattern = "|".join(valid_synonyms)
            mask = candidates['source'].str.contains(pattern, case=False, na=False)
            filtered = candidates[mask]
            results.extend(filtered.to_dict('records'))

    # Step 3: semantic fallback if still fewer than 5
    if len(results) < 5:
        sims = cosine_similarity(model.encode([term]), model.encode(category_subset['source'].tolist())).flatten()
        top_indices = sims.argsort()[::-1][:5]
        fallback = category_subset.iloc[top_indices]
        results.extend(fallback.to_dict('records'))

    # Deduplicate and cap at 5
    seen = set()
    final_results = []
    for r in results:
        if r['source'] not in seen:
            final_results.append(r)
            seen.add(r['source'])
        if len(final_results) == 5:
            break

    # Compute BLEU, METEOR, COMET
    for r in final_results:
        src, tgt = r['source'], r['target']
        src_tokens = word_tokenize(src)
        tgt_tokens = word_tokenize(tgt)

        r['BLEU'] = sentence_bleu([tgt_tokens], src_tokens, smoothing_function=SmoothingFunction().method1)
        r['METEOR'] = meteor_score([tgt_tokens], src_tokens)
        r['COMET'] = comet_model.predict([{"src": src, "mt": tgt, "ref": tgt}]).scores[0]

    return pd.DataFrame(final_results)

# --- Definition fetch ---
def get_definition(term, glossary):
    row = glossary[glossary['legal_term'].str.lower() == term.lower()]
    if not row.empty:
        return row.iloc[0]['definition'], row.iloc[0]['legal_category']
    return " No definition found.", None

# --- Batch processing ---
batch_size = 10
all_results = []

for start in range(0, len(corpus), batch_size):
    batch = corpus.iloc[start:start+batch_size]
    print(f"\n🔹 Processing batch {start}–{start+len(batch)-1}")

    batch_results = []

    for idx, row in batch.iterrows():
        input_sentence = row['source']
        glossary_terms = glossary['legal_term'].dropna().unique()
        legal_terms = [
            t for t in glossary_terms
            if re.search(rf"\b{re.escape(t.lower())}\b", str(input_sentence).lower())
            and t.lower() not in stoplist
        ]

        if not legal_terms:
            continue

        for term in legal_terms:
            definition, category = get_definition(term, glossary)
            if category:
                examples = get_examples_semantic(corpus, term, category)
                if not examples.empty:
                    examples['row_id'] = idx
                    examples['term'] = term
                    batch_results.extend(examples.to_dict('records'))

   
    if batch_results:
        all_results.extend(batch_results)
        batch_df = pd.DataFrame(batch_results)
        batch_filename = f"legal_examples_batch_{start}_{start+len(batch)-1}.csv"
        batch_df = batch_df[['source','target','legal_entity','BLEU','METEOR','COMET','term']]
        batch_df.to_csv(batch_filename, index=False)
        print(f" Saved batch results to {batch_filename}")
    else:
        print(" No legal terms found in this batch")

# --- Save all results to CSV with only required columns ---
results_df = pd.DataFrame(all_results)
results_df = results_df[['source','target','legal_entity','BLEU','METEOR','COMET','term']]
results_df.to_csv("legal_examples_with_metrics.csv", index=False)

print("\n Saved results to legal_examples_with_metrics.csv with selected columns only")
