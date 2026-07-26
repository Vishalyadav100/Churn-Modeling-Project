from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
    
)

import pandas as pd
import time

def get_models():  # Models Dictionary

    models = {

        "Logistic Regression":
            LogisticRegression(max_iter=1000),

        "Decision Tree":
            DecisionTreeClassifier(random_state=42),

        "Random Forest":
            RandomForestClassifier(n_estimators=50,max_depth=10,random_state=42,n_jobs=1),

        "XGBoost":
            XGBClassifier(n_estimators=50,max_depth=5,use_label_encoder=False,eval_metric="logloss")

    }

    return models

def train_models(     # Train Models Function
    models,
    X_train,
    X_test,
    y_train,
    y_test
):

    results = []

    trained_models = {}

    for name, model in models.items(): #
        start = time.time()
        model.fit( X_train,y_train)
        end = time.time()
        prediction = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            prediction
        )

        precision = precision_score(
            y_test,
            prediction,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            prediction,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            prediction,
            average="weighted",
            zero_division=0
        )

        results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "Training Time": round(end-start,4)

    })
 
        trained_models[name] = model

    return results, trained_models

def result_dataframe(results): #Result DataFrame

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(

        by="Accuracy",

        ascending=False

    )

    results_df["Rank"] = range(

        1,

        len(results_df)+1

    )

    return results_df

def best_model(results_df):   #Best Model

    return results_df.iloc[0]["Model"]