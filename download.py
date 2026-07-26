import streamlit as st
import pandas as pd

def download_results(results_df):

    csv = results_df.to_csv(index=False)

    st.download_button(
        label="📄 Download Model Comparison CSV",
        data=csv,
        file_name="Model_Comparison.csv",
        mime="text/csv",
        key="download_csv"
    )

def download_model():

    with open(
        "saved_models/Best_Model.pkl",
        "rb"
    ) as file:

        st.download_button(

            label="⬇ Download Best Model",

            data=file,

            file_name="Best_Model.pkl",

            mime="application/octet-stream",
            key="download_model"
        )

def model_information(results_df):

    best = results_df.loc[
        results_df["Accuracy"].idxmax()
    ]

    st.subheader("🏆 Best Model Information")

    st.write(f"**Model Name :** {best['Model']}")

    st.write(f"**Accuracy :** {best['Accuracy']:.4f}")

    st.write(f"**Precision :** {best['Precision']:.4f}")

    st.write(f"**Recall :** {best['Recall']:.4f}")

    st.write(f"**F1 Score :** {best['F1 Score']:.4f}")

