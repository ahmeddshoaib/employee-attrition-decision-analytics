# Employee Attrition Decision Analytics

This project examines where employee attrition is concentrated, compares three classification models on the same test population and separates high attrition rates from the number of employees affected.

The analysis covers **1,450 employees**, including **279 leavers**, and combines KNIME modelling with a Tableau decision story. The strongest descriptive concentration appeared around overtime, age, role and department. Demonstration data is included so the full decision workflow can be inspected without exposing employee records.

## Business question

An attrition model is useful only if it leads to a responsible intervention. This project therefore moves from prediction to prioritisation: where is attrition unusually concentrated, how many employees are affected, which patterns remain visible across different models, and what should HR investigate at team level?

The unit of action is a workforce segment rather than an individual employee. Overtime, role, department and age bands are used to identify operating conditions worth reviewing; they are not used to label a person as a likely leaver or to support an adverse employment decision.

## What I built

I prepared the data, compared decision tree, random forest and gradient-boosted models, interpreted the main predictors and connected the results to retention recommendations. The repository also applies consistent class treatment across models, exports a segment-level priority table and checks row grain, labels and saved outputs.

## Project evidence

The submitted analysis reported:

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Decision tree | 81.59% | 0.725 |
| Random forest | 86.20% | 0.871 |
| Gradient boosted trees | **87.24%** | **0.895** |

These figures describe the KNIME model comparison. The Python workflow keeps its demonstration-data checks separate and applies the same class-weighting logic across models for a fair comparison.

![Synthetic model comparison](figures/model_comparison.png)

## Decision framework

A high attrition rate can affect very few employees; a large department can create many leavers at a moderate rate. The decision output therefore uses both:

- **risk intensity:** attrition rate within a segment;
- **business exposure:** number of leavers in that segment.

Segments in the high-rate, high-volume quadrant become the first candidates for workload review, management investigation and retention testing.

That distinction changes the management response. A small segment with a high rate may need a focused qualitative investigation; a large segment with a moderate rate may create the greater recruitment, training and continuity cost. The matrix keeps both questions visible.

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
| `scripts/run_analysis.py` | Model run and output generation |
| `outputs/` | Labelled metrics and decision tables |
| `figures/` | Model and intervention visuals |
| `tests/` | Data-grain, label and output checks |

## Management boundary

This analysis identifies associations and supports prioritisation. It does not establish that overtime causes attrition, and it must not be used for adverse employment decisions about individuals. Any live deployment would require fairness review, data-protection controls, employee consultation and intervention testing.

## Author

**Muhammad Ahmed Shoaib**<br>
HR analytics, classification and decision communication.
