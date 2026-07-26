import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def show_model_ranking(results_df): # Model Ranking Function

    ranking = results_df.sort_values(
        by="Accuracy",
        ascending=False
    )

    ranking["Rank"] = range(
        1,
        len(ranking)+1
    )

    st.subheader("🏆 Model Ranking")

    st.dataframe(ranking)

    return ranking

def accuracy_chart(ranking): # Accuracy chart

    st.subheader("Accuracy Comparison")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.barplot(
        data=ranking,
        x="Model",
        y="Accuracy",
        palette="viridis",
        ax=ax
    )

    plt.xticks(rotation=20)
    st.pyplot(fig)

def performance_chart(ranking): # Perfomance chart
    st.subheader("Performance Metrics")
    chart = ranking.set_index("Model")
    st.line_chart(
        chart[

            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ]

        ]

    )

def best_model_card(ranking):  # Best model

    best = ranking.iloc[0]

    st.success(

        f"""

🏆 Best Model : {best['Model']}

Accuracy : {best['Accuracy']:.4f}

Precision : {best['Precision']:.4f}

Recall : {best['Recall']:.4f}

F1 Score : {best['F1 Score']:.4f}

"""
    )

def download_results(ranking): #
    csv = ranking.to_csv(index=False)
   

def accuracy_chart(results_df):

    fig, ax = plt.subplots(figsize=(10,5))

    colors = [
        "royalblue",
        "orange",
        "green",
        "red"
    ]

    bars = ax.bar(
        results_df["Model"],
        results_df["Accuracy"],
        color=colors[:len(results_df)]
    )

    ax.set_title(
        "Model Accuracy Comparison",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_ylabel("Accuracy")

    plt.xticks(rotation=20)

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x()+bar.get_width()/2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom"
        )

    st.pyplot(fig)

def precision_chart(results_df):

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(
        results_df["Model"],
        results_df["Precision"]
    )

    ax.set_title("Precision Comparison")
    ax.set_ylabel("Precision")

    plt.xticks(rotation=20)

    st.pyplot(fig)

def recall_chart(results_df):

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(
        results_df["Model"],
        results_df["Recall"]
    )

    ax.set_title("Recall Comparison")
    ax.set_ylabel("Recall")

    plt.xticks(rotation=20)

    st.pyplot(fig)

def f1_chart(results_df):

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(
        results_df["Model"],
        results_df["F1 Score"]
    )

    ax.set_title("F1 Score Comparison")
    ax.set_ylabel("F1 Score")

    plt.xticks(rotation=20)

    st.pyplot(fig)

def training_time_chart(results_df):

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(
        results_df["Model"],
        results_df["Training Time"]
    )

    ax.set_title("Training Time Comparison")
    ax.set_ylabel("Seconds")

    plt.xticks(rotation=20)

    st.pyplot(fig)