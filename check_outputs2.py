import csv

with open("outputs/memfaith/gpt2_full_context_results.csv") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 5: break
        print(f"Example {row['example_id']}:")
        print(f"  Gold: {row['gold_answer']}")
        print(f"  Extracted: {row['prediction_normalized']}")
        print(f"  Raw: {repr(row['prediction_raw'][:150])}")
        print()
