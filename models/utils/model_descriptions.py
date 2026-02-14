MODEL_DESCRIPTIONS = {
    "Logistic Regression": {
        "summary": "A linear classification model that estimates the probability of a customer subscribing.",
        "pros": [
            "Fast and interpretable",
            "Strong baseline performance",
            "Stable results on imbalanced data"
        ],
        "cons": [
            "Cannot capture complex non-linear patterns",
            "Lower recall for positive (yes) class"
        ],
        "expectation": "High accuracy and AUC, but moderate recall for YES predictions."
    },

    "Decision Tree": {
        "summary": "A rule-based model that splits data using decision rules.",
        "pros": [
            "Easy to interpret",
            "Captures non-linear relationships"
        ],
        "cons": [
            "Prone to overfitting",
            "Less stable on unseen data"
        ],
        "expectation": "Balanced recall and precision but lower overall performance."
    },

    "KNN": {
        "summary": "A distance-based model that classifies customers based on similarity.",
        "pros": [
            "Simple and intuitive",
            "Works well when similar customers behave similarly"
        ],
        "cons": [
            "Slow on large datasets",
            "Sensitive to feature scaling"
        ],
        "expectation": "Moderate performance with limited improvement on minority class."
    },

    "Naive Bayes": {
        "summary": "A probabilistic model based on Bayes’ theorem with feature independence assumptions.",
        "pros": [
            "Very fast",
            "Handles categorical data well"
        ],
        "cons": [
            "Strong independence assumptions",
            "Lower precision"
        ],
        "expectation": "Higher recall for YES cases but reduced precision."
    },

    "Random Forest": {
        "summary": "An ensemble of decision trees that improves stability and accuracy.",
        "pros": [
            "Handles non-linear patterns well",
            "Robust to noise and overfitting"
        ],
        "cons": [
            "Less interpretable",
            "Slightly higher computational cost"
        ],
        "expectation": "Strong overall performance with improved F1 and MCC."
    },

    "XGBoost": {
        "summary": "A gradient boosting ensemble that builds trees sequentially to correct errors.",
        "pros": [
            "Best overall performance",
            "Excellent class separation (high AUC)"
        ],
        "cons": [
            "More complex to tune",
            "Less interpretable"
        ],
        "expectation": "Highest recall and F1-score for YES predictions after threshold tuning."
    }
}
