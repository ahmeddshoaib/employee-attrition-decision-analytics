from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd


class AttritionOutputsTest(unittest.TestCase):
    def test_demo_grain_and_class_count(self) -> None:
        data = pd.read_csv(Path("data/demo_employee_attrition.csv"))
        self.assertEqual(len(data), 1450)
        self.assertEqual((data["Attrition"] == "Yes").sum(), 279)
        self.assertEqual(data["EmployeeID"].nunique(), 1450)

    def test_metrics_boundary(self) -> None:
        metrics = pd.read_csv(Path("outputs/synthetic_model_metrics.csv"))
        self.assertTrue(metrics["data_label"].eq("synthetic_demo").all())
        self.assertEqual(set(metrics["model"]), {"decision_tree", "random_forest", "gradient_boosting"})
        scores = metrics[["f1", "roc_auc"]]
        self.assertTrue(((scores >= 0) & (scores <= 1)).all().all())


if __name__ == "__main__":
    unittest.main()
