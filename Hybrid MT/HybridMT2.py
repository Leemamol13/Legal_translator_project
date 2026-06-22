# -----------------------------
# Save as separate Excel files (250 rows each)
# -----------------------------

output_prefix = "legal_examples_batch"
batch_size = 250

batch_results = []
batch_number = 1
total_rows = 0


for idx, row in corpus.iterrows():

    source = row["source"]
    reference = row["target"]

    terms = []

    for t in glossary["legal_term"].dropna():

        if (
            re.search(
                rf"\b{re.escape(t.lower())}\b",
                source.lower()
            )
            and
            t.lower() not in STOPLIST
        ):
            terms.append(t)


    if not terms:
        continue


    definitions = {}
    all_examples = []


    for term in terms:

        definition, category = get_definition(term)

        if not definition:
            continue


        definitions[term] = definition


        examples = retrieve_examples(
            corpus,
            term,
            category,
            top_k=5
        )


        all_examples.extend(examples)


        translation = translate_hybrid(
            source,
            definitions,
            all_examples[:5]
        )


        bleu, meteor, comet = evaluate(
            source,
            translation,
            reference
        )


        batch_results.append(
            {
                "source": source,
                "reference": reference,
                "term": term,
                "definition_used": definition,
                "retrieved_examples": len(examples),
                "translation": translation,
                "BLEU": bleu,
                "METEOR": meteor,
                "COMET": comet
            }
        )


    # when 250 rows reached
    if len(batch_results) >= batch_size:

        batch_df = pd.DataFrame(
            batch_results[:batch_size]
        )

        file_name = (
            f"{output_prefix}_{batch_number}.xlsx"
        )

        batch_df.to_excel(
            file_name,
            index=False
        )

        print(
            f"Saved {file_name}"
        )


        batch_number += 1

        total_rows += batch_size

        batch_results = batch_results[batch_size:]



# save remaining rows
if batch_results:

    batch_df = pd.DataFrame(batch_results)

    file_name = (
        f"{output_prefix}_{batch_number}.xlsx"
    )

    batch_df.to_excel(
        file_name,
        index=False
    )

    total_rows += len(batch_results)

    print(
        f"Saved {file_name}"
    )


print(
    f"Completed. Total rows saved: {total_rows}"
)