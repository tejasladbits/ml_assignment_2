import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
from matplotlib import pyplot as plt
from models.utils.model_descriptions import MODEL_DESCRIPTIONS
from models.utils.generate_test_data import generate_test_data
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    f1_score, recall_score, matthews_corrcoef,
    confusion_matrix, precision_score
)

if "show_comparison" not in st.session_state:
    st.session_state.show_comparison = False

st.set_page_config(
    page_title="Bank Marketing data analytics",
    page_icon="🦈",
    layout="wide"
)

st.title("ML Assignment - 2 (Dataset: Bank Marketing data analytics)")
st.write("Name: Tejas Kishor Lad")
st.write("BITS ID: 2025AA05206")
st.divider()

@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
        "Decision Tree": joblib.load("models/decision_tree.pkl"),
        "KNN": joblib.load("models/knn.pkl"),
        "Naive Bayes": joblib.load("models/naive_bayes.pkl"),
        "Random Forest": joblib.load("models/random_forest.pkl"),
        "XGBoost": joblib.load("models/xgboost.pkl"),
    }

models = load_models()

@st.cache_data
def load_existing_dataset():
    return pd.read_csv("data/bank_marketing.csv")

df_base = load_existing_dataset()

left_col, divider_col, right_col = st.columns([1.3, 0.05, 2.7])

with left_col:
    with st.expander("Download Sample Test Data"):
        dcol1, dcol2, dcol3 = st.columns(3)

        with dcol1:
            sample_size = st.selectbox("Sample Size", [500, 1000, 2000])

        with dcol2:
            class_balance = st.selectbox(
                "Class Distribution",
                ["Original", "More YES cases"]
            )

        with dcol3:
            random_state = st.number_input(
                "Random Seed",
                min_value=1,
                max_value=9999,
                value=42
            )

        test_df = generate_test_data(
            df_base,
            sample_size,
            class_balance,
            random_state
        )

        csv_data = test_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Test CSV",
            data=csv_data,
            file_name="bank_marketing_test_data.csv",
            mime="text/csv"
        )

    uploaded_file = st.file_uploader(
        "**Upload Test Dataset (CSV)**",
        type=["csv"]
    )

    selected_model_name = st.selectbox(
        "**Select Model**",
        list(models.keys())
    )

    desc = MODEL_DESCRIPTIONS[selected_model_name]
    st.subheader(f"***Model: {selected_model_name} :-***")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Strengths**")
        for p in desc["pros"]:
            st.markdown(f"- {p}")

    with col2:
        st.markdown("**Limitations**")
        for c in desc["cons"]:
            st.markdown(f"- {c}")

    st.info(f"**Expected Results:** {desc['expectation']}")

with divider_col:
    st.markdown(
        """
        <div style="
            height: 100vh;
            border-left: 1px solid #e0e0e0;
            margin: auto;
        "></div>
        """,
        unsafe_allow_html=True
    )

with right_col:
    if st.button("🔁 Compare All Models"):
        st.session_state.show_comparison = not st.session_state.show_comparison

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "y" not in df.columns:
            st.error("Target variable not found")
            st.stop()

        X_test = df.drop("y", axis=1)
        y_test = df["y"].map({"no": 0, "yes": 1})

        if st.session_state.show_comparison:
            results = []

            for model_name, model in models.items():
                y_proba = model.predict_proba(X_test)[:, 1]
                y_pred = (y_proba >= 0.35).astype(int)

                results.append({
                    "Model": model_name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "AUC": roc_auc_score(y_test, y_proba),
                    "Precision": precision_score(y_test, y_pred),
                    "Recall": recall_score(y_test, y_pred),
                    "F1": f1_score(y_test, y_pred),
                    "MCC": matthews_corrcoef(y_test, y_pred)
                })

            st.subheader("Model Comparison")

            results_df = pd.DataFrame(results)
            numeric_cols = results_df.columns.drop("Model")

            st.dataframe(
                results_df.style.format({col: "{:.4f}" for col in numeric_cols}),
                use_container_width=True
            )


        else:
            y_proba = models[selected_model_name].predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.35).astype(int)

            st.subheader("📈 Performance Metrics")

            m1, m2, m3 = st.columns(3)
            m4, m5, m6 = st.columns(3)

            m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            m2.metric("AUC", f"{roc_auc_score(y_test, y_proba):.4f}")
            m3.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")

            m4.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
            m5.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
            m6.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")

            st.subheader("🧩 Confusion Matrix Analysis")

            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="coolwarm",
                xticklabels=["No", "Yes"],
                yticklabels=["No", "Yes"],
                ax=ax
            )

            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_title("Confusion Matrix")

            st.pyplot(fig)

st.divider()