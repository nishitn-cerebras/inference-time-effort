import os
import json
import glob
import statistics
from collections import defaultdict
from tabulate import tabulate

def aggregate_plan_success_rates(jsonl_dir):
    source_to_scores = defaultdict(list)

    for file_path in glob.glob(os.path.join(jsonl_dir, "*.jsonl")):
        with open(file_path, "r") as f:
            try:
                line = f.readline().strip()
                data = json.loads(line)
                source = data["source"]
                plans = data.get("plans_and_executions", [])

                for plan in plans:
                    success_rate = plan.get("plan_success_rate")
                    if success_rate is not None:
                        source_to_scores[source].append(success_rate)

            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

    # Compute mean and stddev
    rows = []
    for source, scores in sorted(source_to_scores.items()):
        mean = sum(scores) / len(scores)
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        rows.append([source, len(scores), round(mean, 3), round(std_dev, 3)])

    print(tabulate(rows, headers=["Source", "Num Plans", "Mean Success Rate", "Std Dev"]))
    return rows

if __name__ == "__main__":
    output_dir = "/cb/cold2/nishitn/mlftmp3_copied/skywork_curator"
    aggregate_plan_success_rates(output_dir)
