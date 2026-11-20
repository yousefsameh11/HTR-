# Report

Technical report and presentation slides for the Handwritten Text Recognition project.

## Contents

- `report.pdf` — full technical report (22 sections: abstract through future work)
- `slides.pdf` — presentation slides

## Report structure (per project spec §4.3)

1. Abstract
2. Introduction
3. Problem Statement
4. Motivation
5. Related Work
6. Dataset
7. Data Preprocessing
8. Methodology
9. Baseline
10. Proposed/Advanced Models
11. Experimental Setup
12. Hyperparameters
13. Results
14. Model Comparison
15. Ablation Study
16. Error Analysis
17. Discussion
18. Limitations
19. Ethical Considerations
20. Conclusion
21. Future Work
22. References

## Required tables (per project spec §4.4)

1. Dataset comparison
2. Model architecture comparison
3. Hyperparameter configuration
4. Main experimental results
5. Ablation study
6. Error analysis
7. Computational cost

## Real results used in the report

All numbers in the report come from the trained checkpoints and evaluation artifacts
in `experiments/`. No metrics are fabricated.

| Model | CER | WER | Speed | Training |
|-------|-----|-----|-------|----------|
| CRNN + BiLSTM + CTC | 14.06 % | 43.51 % | 0.6 ms/line | 49 epochs, ~53 min RTX 3050 |
| TrOCR-small fine-tuned | 5.34 % | 10.88 % | 363 ms/line | 5 epochs, 14.3 min RTX 3050 |
