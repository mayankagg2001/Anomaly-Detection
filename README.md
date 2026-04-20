# Credit Card Fraud Detection

This notebook demonstrates various machine learning techniques for detecting fraudulent credit card transactions. The primary goal is to build models that can accurately identify rare fraudulent transactions while minimizing false positives.

## Dataset

The dataset used in this analysis is the **'Credit Card Fraud Detection'** dataset (`mlg-ulb/creditcardfraud`) from Kaggle. It contains anonymized credit card transaction data, where each row represents a transaction. The dataset features include:

- `Time`: Seconds elapsed between this transaction and the first transaction in the dataset.
- `V1` - `V28`: Principal components obtained with PCA, representing anonymized features due to confidentiality issues.
- `Amount`: Transaction Amount.
- `Class`: Response variable, which is `1` for fraudulent transactions and `0` otherwise.

**Dataset Overview**:
- Total Transactions: 284,807
- Normal Transactions (Class 0): 284,315
- Fraudulent Transactions (Class 1): 492
- Contamination Rate (approximate anomaly proportion): ~0.17%

## Methodology

The notebook follows a comprehensive approach to anomaly detection and supervised classification, including:

1.  **Kaggle API Setup**: Configuration of Kaggle API credentials using Colab secrets for secure dataset access.
2.  **Data Loading and Initial Exploration**: Loading the `creditcard.csv` into a pandas DataFrame and performing initial checks (`df.shape`, `df.head()`).
3.  **Initial Anomaly Detection**: 
    - **Isolation Forest**: An unsupervised anomaly detection algorithm trained on the entire dataset.
    - **Gaussian Probability Method**: An anomaly detection technique based on estimating Gaussian parameters from normal data and calculating log probabilities.
4.  **Feature Selection**: Identifying the top 10 features most correlated with the 'Class' column to potentially improve model performance and reduce noise.
5.  **Feature Scaling**: Applying `StandardScaler` to normalize numerical features, ensuring all features contribute equally to the models.
6.  **Gaussian Probability with Selected Features**: Re-evaluating the Gaussian Probability method using only the selected, scaled features.
7.  **Supervised Learning Approaches**: Leveraging the labeled data to train more powerful classification models.
    - **XGBoost Classifier**: Trained with `scale_pos_weight` to handle class imbalance.
    - **SMOTE (Synthetic Minority Over-sampling Technique)**: Used to generate synthetic samples for the minority class, addressing class imbalance.
    - **XGBoost with SMOTE**: Training XGBoost on SMOTE-resampled data.
    - **Random Forest with SMOTE**: Training a Random Forest classifier on SMOTE-resampled data.
8.  **Hyperparameter Tuning (Random Forest)**: Using `GridSearchCV` to optimize Random Forest parameters (`n_estimators`, `max_depth`, `min_samples_leaf`) for improved performance.
9.  **Tuning SMOTE Sampling Strategy**: Exploring different `sampling_strategy` ratios for SMOTE within a `Pipeline` and `GridSearchCV` to find the optimal balance for the minority class.
10. **Hybrid Approach (Isolation Forest Scores as Features)**: Incorporating anomaly scores from Isolation Forest as an additional feature for the Random Forest model, combining unsupervised and supervised learning strengths.
11. **Threshold Optimization**: Visualizing Precision-Recall curves to understand trade-offs and guide the selection of an optimal classification threshold based on business needs.

## Model Performance Comparison

The table below summarizes the performance of various models and strategies, focusing on Precision, Recall, and F1-Score for the minority class (fraudulent transactions):

| Model | Strategy | Precision | Recall | F1-Score | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Isolation Forest** | Unsupervised (All Features) | 0.26 | 0.26 | 0.26 | Baseline unsupervised performance. |
| **Gaussian Prob.** | Unsupervised (Top 5 Features) | 0.43 | 0.76 | 0.55 | Significant boost from feature selection. |
| **XGBoost** | Supervised (Weighted) | 0.33 | 0.87 | 0.47 | High recall, but many false positives. |
| **Optimized RF** | Supervised + Tuning | 0.66 | 0.85 | 0.74 | Random Forest with tuned depth and leaf parameters. |
| **RF + Tuned SMOTE** | **Supervised (SMOTE Sampling Strategy 0.1)** | **0.86** | **0.86** | **0.86** | **Best performing model overall, achieving a balanced F1-score.** |
| **Hybrid Model** | RF + SMOTE + Iso-Score | 0.84 | 0.86 | 0.85 | Confirmed that the tuned RF already captured outlier patterns effectively. |

## Key Insights

-   **Optimal Oversampling Ratio**: A crucial finding was that balancing classes to 50/50 (default SMOTE) was less effective than a more moderate 10% ratio for the minority class. This prevented overfitting and significantly boosted Precision.
-   **Feature Engineering vs. Hybridization**: While the Hybrid model (RF + SMOTE + Isolation Forest scores) showed strong performance, the well-tuned Random Forest with optimized SMOTE parameters performed comparably, suggesting that for this dataset, the Random Forest could extract similar insights from the raw features.
-   **Precision-Recall Balance**: The best model achieved a stable balance of 86% across both Precision and Recall, indicating a robust ability to detect fraud while keeping false alarms manageable.

## Final Recommendation

The **Random Forest model with a Tuned SMOTE sampling strategy (0.1)** is recommended. It provides a high and balanced precision and recall, offering a strong solution for credit card fraud detection with good generalization capabilities.

## Strategies for Further Improvement

To further enhance model performance, the following advanced techniques can be considered:

1.  **Feature Engineering**: Explore interaction terms (e.g., `V17 * V14`) and time-based features (e.g., 'Hour of Day').
2.  **Threshold Optimization**: Utilize Precision-Recall Curves to fine-tune the classification threshold based on specific business priorities (e.g., prioritizing recall to catch more fraud).
3.  **Cost-Sensitive Learning**: Implement `sample_weight` in the Random Forest to assign higher penalties for false negatives (missing fraud).
4.  **Ensembling (Stacking)**: Combine predictions from multiple models (e.g., Random Forest, XGBoost, k-Nearest Neighbors) using a meta-learner.
5.  **External Data**: Incorporate additional data sources such as geographical information or user behavior history if available.
