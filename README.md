# ML Assignment - 2
## Bank Marketing Data Analytics using Machine Learning


**Author:** Tejas Kishor Lad
**BITS ID:** 2025AA05206

---

## a. Problem Statement

The objective of this assignment is to analyze the Bank Marketing dataset and build machine learning models to predict whether a customer will subscribe to a term deposit. The task is to predict the target with "yes" or "no".

## b. Dataset Description

- Dataset: Bank Marketing Dataset
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- Target variable: `y`
  - `yes`: Customer subscribes to the term deposit
  - `no`: Customer does not subscribe

The dataset is imbalanced, with a smaller number of occurances of`yes` cases as compared to `no` cases.

##### Data Preprocessing:

The following preprocessing steps were applied on the data:

- Separation of features and target variable (`y`) 
- Encoding of target variable (`yes` = 1, `no` = 0)
- Handling missing values:
  - "Median" imputation for numerical features
  - "Most frequent value" imputation for categorical features
- One - Hot Encoding for categorical variables
- Feature scaling wherever required

## c. Models Used

The following classification models were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Random Forest
6. XGBoost

All models are implemented in Python files and trained using the same preprocessing pipeline.

##### Model Saving and Artifacts

- Model implementation and training logic are written in `train_models.py` file
- Trained models are serialized using `joblib` and stored as `[model_name].pkl` files
- These serialized models are loaded in the Streamlit application

##### Evaluation Metrics

Each model is evaluated using the following metrics:

- Accuracy
- AUC (Area Under the ROC Curve)
- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient (MCC)

##### Comparison Table with the evaluation metrics calculated for all the 6 models:

| Model                | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|---------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression | 0.9019   | 0.9006 | 0.6049    | 0.4660 | 0.5264 | 0.4778 |
| Decision Tree       | 0.8713   | 0.6971 | 0.4518    | 0.4698 | 0.4606 | 0.3877 |
| KNN                 | 0.8883   | 0.8183 | 0.5215    | 0.5510 | 0.5358 | 0.4727 |
| Naive Bayes         | 0.8491   | 0.8020 | 0.3887    | 0.5066 | 0.4399 | 0.3586 |
| Random Forest       | 0.8986   | 0.9184 | 0.5580    | 0.6408 | 0.5966 | 0.5406 |
| XGBoost             | 0.9036   | 0.9227 | 0.5826    | 0.6200 | 0.6007 | 0.5463 |

##### Observations on the performance of each model on the dataset:

| ML Model Name | Observation about model performance |
|--------------|--------------------------------------|
| Logistic Regression | Logistic Regression provides a strong baseline model with high accuracy and AUC, indicating good overall class separation. However, recall for the positive (yes) class is less than 0.5, which means there were more number of missed potential subscribers in an imbalanced dataset. |
| Decision Tree | The Decision Tree model shows lower AUC and overall performance compared to other models. It achieves balanced precision and recall, it is possible to get overfit and does not generalize well on this dataset. |
| kNN | The kNN model results in above acceptable recall but the precion is lower. Its performance is sensitive to feature scaling and distance calculations, which affects robustness on a high-dimensional data. |
| Naive Bayes | Naive Bayes achieves higher recall relative to precision, meaning it identifies more positive cases but it increased false positives. This behavior comes from its feature independence assumption. |
| Random Forest (Ensemble) | Random Forest improves performance by combining multiple trees, and thus reducing overfitting. It achieves a better balanced between precision and recall, resulting in higher F1-score and MCC. |
| XGBoost (Ensemble) | XGBoost provides the best overall performance with the highest AUC, F1-score, and MCC. It captures complex relationships in the data and handles class imbalance effectively. |


Observations from the comparison indicate that ensemble models perform better than linear and distance-based models on this dataset.

##### Threshold Tuning

Instead of relying only on the default prediction threshold of 0.5, a lower threshold (0.35) improves recall, F1-score, and MCC, with a small reduction in accuracy and its a considerable tradeoff.

This is suitable for marketing analytics, where missing potential subscribers is costlier than contacting non-subscribers.

## d. Streamlit Application
URL: https://tejas-lad-ml-assignment.streamlit.app/
An interactive Streamlit application was developed with the following features:

- Download sample test datasets with configurable parameters
- Upload test dataset in CSV format
- Select a trained classification model from the dropdown
- View model description including the model strengths and limitations
- Display evaluation metrics for the selected model
- Visualize confusion matrix for the selected model and test data
- View comparison of metrics for all the models

---