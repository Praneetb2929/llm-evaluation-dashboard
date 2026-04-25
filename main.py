import json
import pandas as pd
from model_runner import run_all
from scorer import score_all

# Load dataset
with open("dataset.json") as f:
    dataset = json.load(f)

# Step 1: collect model responses
print("Running models...")
results = run_all(dataset)

# Step 2: score each response
print("Scoring responses...")
results = score_all(results)

# Step 3: save to CSV for the dashboard
df = pd.DataFrame(results)
df.to_csv("eval_results.csv", index=False)
print("Done! Results saved to eval_results.csv")