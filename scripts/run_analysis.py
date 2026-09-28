"""Run the public employee attrition decision workflow."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.attrition import evaluate_models, load_data, retention_priority, save_figures  # noqa: E402


data = load_data(ROOT / "data" / "demo_employee_attrition.csv")
metrics, scores = evaluate_models(data)
priority = retention_priority(data)

output = ROOT / "outputs"
output.mkdir(exist_ok=True)
metrics.to_csv(output / "synthetic_model_metrics.csv", index=False)
scores.to_csv(output / "synthetic_test_scores.csv", index=False)
priority.to_csv(output / "synthetic_retention_priority.csv", index=False)
save_figures(metrics, priority, ROOT / "figures")

print(metrics.to_string(index=False))
print("ATTRITION ANALYSIS COMPLETED")

