"""Generate a synthetic HR dataset matching the academic case grain."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RNG = np.random.default_rng(42)
N = 1450

department = RNG.choice(["Research & Development", "Sales", "Human Resources"], N, p=[0.65, 0.30, 0.05])
role_by_department = {
    "Research & Development": ["Research Scientist", "Laboratory Technician", "Manufacturing Director", "Research Director"],
    "Sales": ["Sales Executive", "Sales Representative", "Manager"],
    "Human Resources": ["Human Resources", "Manager"],
}
job_role = np.array([RNG.choice(role_by_department[value]) for value in department])
age = np.clip(np.rint(RNG.normal(37, 9, N)), 18, 65).astype(int)
overtime = RNG.choice(["Yes", "No"], N, p=[0.29, 0.71])
job_level = np.clip(np.rint((age - 18) / 12 + RNG.normal(0, 0.8, N)), 1, 5).astype(int)
total_years = np.clip(age - 18 - RNG.integers(0, 8, N), 0, None)
years_company = np.array([RNG.integers(0, max(1, value + 1)) for value in total_years])
job_satisfaction = RNG.integers(1, 5, N)
work_life = RNG.integers(1, 5, N)
relationship = RNG.integers(1, 5, N)
bonus = np.where(department == "Sales", RNG.binomial(1, 0.52, N), RNG.binomial(1, 0.18, N))

logit = (
    -2.2
    + 1.25 * (overtime == "Yes")
    + 0.55 * (job_level == 1)
    + 0.45 * np.isin(job_role, ["Sales Representative", "Laboratory Technician"])
    - 0.28 * (job_satisfaction - 2.5)
    - 0.20 * (work_life - 2.5)
    - 0.015 * (age - 35)
    - 0.30 * bonus
)
probability = 1 / (1 + np.exp(-logit))
raw = probability + RNG.normal(0, 0.035, N)
leaver_indices = np.argsort(raw)[-279:]
attrition = np.zeros(N, dtype=int)
attrition[leaver_indices] = 1

frame = pd.DataFrame(
    {
        "EmployeeID": np.arange(50001, 50001 + N),
        "Age": age,
        "Department": department,
        "JobRole": job_role,
        "OverTime": overtime,
        "JobLevel": job_level,
        "MonthlyIncome": np.clip(1700 + job_level * 1750 + RNG.normal(0, 900, N), 1200, None).round(0),
        "TotalWorkingYears": total_years,
        "YearsAtCompany": years_company,
        "JobSatisfaction": job_satisfaction,
        "WorkLifeBalance": work_life,
        "RelationshipSatisfaction": relationship,
        "BonusReceived": bonus,
        "Attrition": np.where(attrition == 1, "Yes", "No"),
    }
)

path = Path("data/demo_employee_attrition.csv")
path.parent.mkdir(parents=True, exist_ok=True)
frame.to_csv(path, index=False)
print(f"Wrote {len(frame):,} synthetic employees with {attrition.sum():,} leavers")

