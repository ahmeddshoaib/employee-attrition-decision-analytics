# Employee Attrition Decision Analytics

A reproducible employee-retention case that combines segment diagnosis, comparable model evaluation and a decision matrix for prioritising interventions.

The submitted MSc project used KNIME and Tableau on **1,450 employees**, including **279 leavers**. It found the strongest descriptive concentration around overtime, age, role and department. The public rebuild keeps that decision logic while replacing the unavailable university dataset with a clearly labelled synthetic validation set.

## Historical project evidence

The submitted analysis reported:

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Decision tree | 81.59% | 0.725 |
| Random forest | 86.20% | 0.871 |
| Gradient boosted trees | **87.24%** | **0.895** |

These figures describe the archived KNIME run. They are not presented as results of this public rebuild. The new pipeline reports its synthetic check separately and applies the same class-weighting logic across models for a fairer comparison.

![Synthetic model comparison](figures/model_comparison.png)

## Decision framework

A high attrition rate can affect very few employees; a large department can create many leavers at a moderate rate. The decision output therefore uses both:

- **risk intensity:** attrition rate within a segment;
- **business exposure:** number of leavers in that segment.

Segments in the high-rate, high-volume quadrant become the first candidates for workload review, management investigation and retention testing.

![Synthetic priority matrix](figures/retention_priority_matrix.png)

## Evaluation design

- Stratified 80/20 train/test split.
- Preprocessing fitted inside each model pipeline.
- Identical test population for decision tree, random forest and gradient boosting.
- Class weights applied consistently instead of balancing only one model.
- Precision, recall, F1 and ROC-AUC reported alongside accuracy.
- Synthetic and historical results kept in separate tables.

## Repository guide

| Path | Purpose |
|---|---|
| `src/attrition.py` | Preparation, model pipelines, evaluation and priority matrix |
| `scripts/generate_demo_data.py` | Reproducible 1,450-row synthetic HR dataset |
| `scripts/run_analysis.py` | End-to-end run |
| `outputs/` | Labelled metrics and decision tables |
| `figures/` | Model and intervention visuals |
| `tests/` | Data-grain, label and output checks |

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/generate_demo_data.py
python scripts/run_analysis.py
python -m unittest discover -s tests -v
```

## Management boundary

This analysis identifies associations and supports prioritisation. It does not establish that overtime causes attrition, and it must not be used for adverse employment decisions about individuals. Any live deployment would require fairness review, data-protection controls, employee consultation and intervention testing.

## Author

**Muhammad Ahmed Shoaib** — HR analytics, classification and decision communication.

