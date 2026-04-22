# Summary of Anomaly Detection Results

## 1. Dataset Overview
- **Total Transactions**: 284,807
- **Normal Transactions (Class 0)**: 284,315
- **Fraudulent Transactions (Class 1)**: 492
- **Contamination Rate**: ~0.17%

## 2. Model Performance Comparison

| Model | Strategy | Precision | Recall | F1-Score | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Isolation Forest** | Unsupervised (All Features) | 0.26 | 0.26 | 0.26 | Baseline unsupervised performance. |
| **Gaussian Prob.** | Unsupervised (Top 5 Features) | 0.43 | 0.76 | 0.55 | Significant boost from feature selection. |
| **XGBoost** | Supervised (Weighted) | 0.33 | 0.87 | 0.47 | High recall, but many false positives. |
| **RF + Tuned SMOTE** | Sampling Strategy (0.1) | 0.86 | 0.86 | 0.86 | Significant improvement using SMOTE. |
| **Hybrid Model** | RF + SMOTE + Iso-Score | 0.84 | 0.86 | 0.85 | Confirmed RF already captured outlier patterns. |
| **SMOTE + Cost-RF** | **Weight Optimization (10:1)** | **0.91** | **0.88** | **0.89** | **Optimized peak: Best balance of FP suppression.** |

## 3. Key Insights
- **Weight Optimization**: Our sweep (testing 1:1 to 20:1) showed that precision peaks at a 10:1 ratio. Beyond this, we see diminishing returns in F1-score as recall stabilizes.
- **The Power of Combination**: Combining SMOTE (to address recall) with Cost-Sensitive weights (to address precision) produced the most robust model.
- **Precision-Recall Optimization**: By assigning a 10:1 weight favoring Class 0, we suppressed false positives that SMOTE might otherwise introduce, achieving ~91% precision in the optimized sweep.
- **Final Recommendation**: Use the **SMOTE + Cost-Sensitive Random Forest (10:1 Ratio)**. It provides the highest F1-Score (0.89) and the most reliable balance for a production fraud detection system.
