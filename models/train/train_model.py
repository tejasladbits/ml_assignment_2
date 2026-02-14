import pandas as pd
import joblib
import os
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder , StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from collections import Counter

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

#  loading the dataset
def load_dataset():
    print("Loading dataset.")
    data_path = "./data/bank_marketing.csv"

    if os.path.exists(data_path):
        print("loading dataset from the csv file")
        df = pd.read_csv(data_path)
    else :
        print("fetching data from uci")
        bank_marketing = fetch_ucirepo(id=222) 
    
        X = bank_marketing.data.features 
        y = bank_marketing.data.targets["y"]

        df = X.copy()
        df["y"] = y

        os.makedirs("data", exist_ok=True)
        df.to_csv(data_path, index=False)
        print("Dataset saved locally.\n\n")

    X = df.drop("y", axis=1)
    y = df["y"]

    print(y.shape)
    print(df.describe)
    print(" Freuqency of target data:-------- ",y.value_counts())
    return X, y

# mapping target yes to 1 and no to 0
def encode_target(y):
    print("Target encoded to 1 and 0.")
    return y.map({'yes': 1, 'no': 0})

# splitting data into test and train dataset
def split_data(X, y):
    print("Data splitting with test size = 0.2")
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing the numeric and categorical columns of the data using columntransformer
def get_preprocessor(X):
    print("Preprocessing data and encoding categorical features")
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='median')),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("encoder", OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor

# training using logistic regression
def train_logistic_regression_model(X_train, y_train, preprocessor):
    max_iterations = 1000
    print(f"Training logistic regression model with iterations: {max_iterations}")
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=max_iterations))
        ]
    )

    model.fit(X_train, y_train)

    print(f"Model training completed using Logistic Regression.")
    return model

# training using decision tree
def train_decision_tree_model(X_train, y_train, preprocessor):
    random_state = 42
    print(f"Training Decision tree classifier model with random state: {random_state}.")
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=random_state))
        ]
    )

    model.fit(X_train, y_train)
    print(f"Model training completed using Decision Tree Classifier.")
    return model

# training using KNN
def train_knn(X_train, y_train, preprocessor):
    n_neighbours=5
    print(f"Training KNN classifier model with n_neighbours: {n_neighbours}.")
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", KNeighborsClassifier(n_neighbors=n_neighbours))
        ]
    )

    model.fit(X_train, y_train)
    print(f"Model training completed using KNN.")
    return model

# training using naive bayes
def train_naive_bayes(X_train, y_train):
    # Naive Bayes must not use the standardScalar
    # Using minmaxScaler for naive bayes
    print(f"Training naive baayes model.")
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    numeric_data_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ])

    categorical_data_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_data_transformer, numeric_features),
            ("cat", categorical_data_transformer, categorical_features)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", GaussianNB())
    ])
    model.fit(X_train, y_train)
    print(f"Model training completed using Naive bayes.")
    return model

# training using random forest
def train_random_forest(X_train, y_train, preprocessor):
    n_estimators=100
    random_state=42
    n_jobs=1

    print(f""" Training naive baayes model with {n_estimators} estimators and 
          random_state: ${random_state}. """)

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-n_jobs,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)
    print(f"Model training completed using random forest.")
    return model

# training using xgboost
def train_xgboost_model(X_train, y_train, preprocessor):
    eval_metrics = "logloss"
    print(f"Training xgBoost model with {eval_metrics} evaluation metrics.")
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            eval_metric=eval_metrics,
            random_state=42,
            use_label_encoder=False
        ))
    ])

    model.fit(X_train, y_train)
    print(f"Model training completed using xgBoost.")
    return model

# evaluating model with the test data and generting evaluation metrics
def evaluate_model(model, X_test, y_test):
    print("Evaluating model on the test data.")
    # y_pred = model.predict(X_test)
    # y_pred_probab = model.predict_proba(X_test)[:, 1]

    y_pred_probab = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_probab >= 0.35).astype(int)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_pred_probab),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "Matthews Correlation Coeﬃcient": matthews_corrcoef(y_test, y_pred)
    }

# saving model to pkl files
def save_model(model, filename):
    print("Saving model to a pkl file")
    joblib.dump(model, filename)

# create test data 
def create_test_data():
    # from your training code
    df_test = X_test.copy()
    df_test["y"] = y_test.map({0: "no", 1: "yes"})
    df_test.to_csv("test_data.csv", index=False)

if __name__=="__main__":
    X, y = load_dataset()
    y = encode_target(y)

    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = get_preprocessor(X_train)

    results = {}

    logistic_regression_model = train_logistic_regression_model(X_train, y_train, preprocessor)
    results["Logistic Regression"] = evaluate_model(logistic_regression_model, X_test, y_test)
    save_model(logistic_regression_model, 'models/logistic_regression.pkl')

    decision_tree_model = train_decision_tree_model(X_train, y_train, preprocessor)
    results["Decision Tree"] = evaluate_model(decision_tree_model, X_test, y_test)
    save_model(decision_tree_model, "models/decision_tree.pkl")

    knn_model = train_knn(X_train, y_train, preprocessor)
    results["KNN"] = evaluate_model(knn_model, X_test, y_test)
    save_model(knn_model, "models/knn.pkl")

    naive_bayes_model = train_naive_bayes(X_train, y_train)
    results["Naive Bayes"] = evaluate_model(naive_bayes_model, X_test, y_test)
    save_model(naive_bayes_model, "models/naive_bayes.pkl")

    random_forest_model = train_random_forest(X_train, y_train, preprocessor)
    results["Random Forest"] = evaluate_model(random_forest_model, X_test, y_test)
    save_model(random_forest_model, "models/random_forest.pkl")

    xgb_model = train_xgboost_model(X_train, y_train, preprocessor)
    results["XGBoost"] = evaluate_model(xgb_model, X_test, y_test)
    save_model(xgb_model, "models/xgboost.pkl")

    print("\nMODEL COMPARISON RESULTS\n")
    for model_name, metrics in results.items():
        print(model_name)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print("-" * 30)

    # create_test_data()

