import json
import numpy as np
from pathlib import Path

def generate_latex_results(trials, name):
    sorted_trials = sorted(trials, key=lambda x: x["value"])
    top5 = sorted_trials[:5]
    best_value = sorted_trials[0]["value"]
    top5_values = [t["value"] for t in top5]
    mean_top5 = np.mean(top5_values)
    std_top5 = np.std(top5_values)

    lines = [
        f"\\textbf{{{name}}} \\\\",
        f"Najlepszy wynik (loss): & {best_value:.4f} \\\\",
        f"Średnia z top-5: & {mean_top5:.4f} \\\\",
        f"Odch. std top-5: & {std_top5:.4f} \\\\",
        "\\hline"
    ]
    return "\n".join(lines)

paths = {
    "Próba 1 (ogólna)": "preprocessing_tuning_first_stage/top_trials.json",
    "Próba 2 (zawężona)": "preprocessing_tuning_second_stage/top_trials.json"
}

output_dir = Path("reports/")
output_dir.mkdir(parents=True, exist_ok=True)

all_latex_blocks = []
for name, path in paths.items():
    with open(path, "r") as f:
        trials = json.load(f)
    latex_block = generate_latex_results(trials, name)
    all_latex_blocks.append(latex_block)

final_output = "\n\n".join(all_latex_blocks)
output_file = output_dir / "results_top5.tex"
output_file.write_text(final_output)

print(f"Results written to {output_file}")