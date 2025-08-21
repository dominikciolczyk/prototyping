import json

# path to your results file
input_file = "results/kd/student_kd_results.jsonl"

# load all records
records = []
with open(input_file, "r") as f:
    for line in f:
        line = line.strip()
        if line:  # skip empty lines
            records.append(json.loads(line))

# sort by score (descending: best first)
records_sorted = sorted(records, key=lambda x: x["score"], reverse=False)

# print top 10 results
for i, rec in enumerate(records_sorted[:10], start=1):
    print(f"{i:02d}. run={rec['run']} | score={rec['score']:.6f}")
