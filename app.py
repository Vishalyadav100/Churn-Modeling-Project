import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from preprocessing import (fill_missing_values,remove_duplicates,
    encode_data,split_features_target,split_data,scale_data)
from preprocessing import (fill_missing_values,
    remove_duplicates,
    encode_data,
    split_features_target,
    split_data,
    scale_data,
    preprocess_pipeline)

from models import (get_models,train_models,
    result_dataframe,
    best_model)

from visualization import (
    show_model_ranking,
    accuracy_chart,
    performance_chart,
    best_model_card,
    download_results
)


from prediction import (
    customer_input,
    encode_prediction,
    load_model,
    predict_customer,
    show_prediction,
    show_probability
)
from visualization import accuracy_chart
from visualization import (
    accuracy_chart,
    precision_chart,
    recall_chart,
    f1_chart,
    training_time_chart
)
from download import (
    download_results,
    download_model,
    model_information
)

st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)
st.sidebar.title("⚙ Settings")

st.sidebar.info("""
Customer Churn Prediction Dashboard

Developed using

✔ Python

✔ Streamlit

✔ Scikit-Learn

✔ XGBoost
""")


st.title("📊 Customer Churn Prediction Dashboard")

st.write(
    "Compare Machine Learning models and predict customer churn."
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "🏠 Home",
    "📂 Dataset Analysis",
    "🧹 Data Preprocessing",
    "🤖 Model Training",
    "📊 Model Comparison",
    "🔮 Customer Prediction",
    "📈 Visualization",
    "⬇ Download Model",
    "ℹ About"
])

@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\project 2\Churn_Modelling.csv")
df = load_data()

with tab1:

    st.header("🏠 Home")

    st.write("""
    Welcome to the Customer Churn Prediction Dashboard.

    This application compares multiple Machine Learning
    algorithms and predicts whether a customer is likely
    to leave the bank.
    """)

    st.subheader("Algorithms Used")

    st.markdown("""
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - XGBoost
    """)

with tab2:

    st.header("📂 Dataset Analysis")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    with col4:
        st.metric("Duplicate Rows", df.duplicated().sum())

    if st.checkbox("Columns Name"):
        st.subheader("Column Names")
        st.write(list(df.columns))    

    if st.checkbox("Data Types"):
        st.subheader("Data Types")
        dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.values
    })
        st.dataframe(dtype_df)

    if st.checkbox("Show missing value"):
        st.subheader("Missing Values")
        missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values
    })
        st.dataframe(missing_df)

    st.subheader("Duplicate Rows")
    st.write("Total Duplicate Rows :", df.duplicated().sum())

    if st.checkbox("Statical Summary"):
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())

    if st.checkbox("show target distribution"):
        st.subheader("Target Distribution")
        st.bar_chart(df["Exited"].value_counts(20))
    if st.checkbox("Correlation Heatmap"):
       st.subheader("Correlation Heatmap")
       fig, ax = plt.subplots(figsize=(10,6))
       sns.heatmap(df.select_dtypes(include=np.number).corr(),
       annot=True,cmap="coolwarm",fmt=".2f",ax=ax)

       st.pyplot(fig)
       plt.close(fig)

    if st.checkbox(" correlation with target"):
        st.subheader("Correlation with Target")

        correlation = df.corr(numeric_only=True)["Exited"].sort_values(
        ascending=False)
        st.bar_chart(correlation)
    if st.checkbox("Numerical Feature Distribution"):
        st.subheader("Numeric Feature Distribution")
        numeric_columns = df.select_dtypes(include=np.number).columns
        selected_numeric = st.selectbox(
       "Select Numeric Column",
        numeric_columns)

        fig, ax = plt.subplots(figsize=(8,4))
        sns.histplot(df[selected_numeric],kde=True,ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    if st.checkbox("Categorical Feature Distribution"):

       st.subheader("Categorical Feature Distribution")
       categorical_columns = df.select_dtypes(include="object").columns

       selected_category = st.selectbox(
       "Select Categorical Column",
        categorical_columns)
       st.bar_chart(
       df[selected_category].value_counts()
)
    
    if st.checkbox("Show Complete Dataset"):
        st.dataframe(df)

    if st.checkbox(" Dataset Sample"):
        st.subheader("Random Sample")
        sample_size = st.slider(
        "Select Number of Rows",5,50,10)
        st.dataframe(df.sample(sample_size))

    if st.checkbox("Download dataset"):
        csv = df.to_csv(index=False)
        

with tab3:    
    st.header("🧹 Data Preprocessing")

    # if st.button("Fill Missing Values"):
    df = fill_missing_values(df)
    st.success("Missing Values Filled Successfully")


    # if st.button("Remove Duplicates"):
    df = remove_duplicates(df)
    st.success("Duplicate Rows Removed")

    
    df = encode_data(df)
    st.success("Encoding Completed")


    # Target Column
    if "Exited" not in df.columns:
      st.error("Target column 'Exited' not found in dataset.")
      st.stop()

    target = "Exited"

    st.success(f"Target Column: {target}")

    # split feature
    test_size = st.slider("Test Size",0.1,0.5,0.2)
    random_state = st.number_input("Random State",value=42)

    # if st.button("🧹 Run Complete Preprocessing"):
    with st.spinner("Preprocessing Dataset..."):
            (processed_df,X_train,X_test,y_train,y_test,feature_names) = preprocess_pipeline(
             df,target,test_size,random_state)




    st.success("Preprocessing Completed Successfully")
    col1, col2 = st.columns(2)   # dataset shape
    with col1:
        st.metric(
        "Training Samples",
        len(X_train))

    with col2:
        st.metric(
        "Testing Samples",
        len(X_test))

    st.metric("Total Features",len(feature_names)) # feature count

    st.subheader("Processed Dataset")
    st.dataframe(processed_df.head())
 
           
with tab4:
    st.header("🤖 Model Training")

    if st.button("Train Models"):
       with st.spinner("Training Models..."):

        models = get_models()

       results, trained_models = train_models(models,X_train,X_test,y_train,y_test)

       results_df = result_dataframe(results)
       st.session_state["results_df"] = results_df

       st.success("All Models Trained Successfully")

       st.subheader("Training Results")
       st.dataframe(results_df)

       best_model_name = best_model(results_df)

       import os
       import joblib
       os.makedirs("saved_models", exist_ok=True)

       joblib.dump(
        trained_models[best_model_name],
        "saved_models/Best_Model.pkl")
       
       st.success("Best Model Saved Successfully")
       
       st.success(f"Best Model : {best_model_name}")


with tab5:
    st.header("📊 Model Comparison")

    if "results_df" not in st.session_state:
       st.warning("Please train the models first.")

    else:
      results_df = st.session_state["results_df"]
      ranking = show_model_ranking(results_df)
      accuracy_chart(ranking)
      performance_chart(ranking)
      best_model_card(ranking)
      

with tab6:
    st.header("🔮 Customer Prediction")
    input_df = customer_input()

    input_df = encode_prediction(input_df)
    st.subheader("Customer Details")
    st.dataframe(input_df)

    if st.button("Predict Customer"):
        with st.spinner("Predicting..."):

         model = load_model()
         prediction, probability = predict_customer(model,input_df)


         show_prediction(prediction)
         show_probability(probability)

with tab7:

    st.header("📈 Visualization")

    if "results_df" in st.session_state:
        
        results_df = st.session_state["results_df"]

        best = results_df.loc[
        results_df["Accuracy"].idxmax()
]
        st.success(f"""🏆 Best Model : {best['Model']}

        Accuracy : {best['Accuracy']:.4f}
"""
)       
        col1,col2,col3,col4 = st.columns(4)

        col1.metric("Best Accuracy",round(results_df["Accuracy"].max(),4))

        col2.metric("Best Precision",round(results_df["Precision"].max(),4))

        col3.metric("Best Recall",round(results_df["Recall"].max(),4))

        col4.metric("Best F1 Score",round(results_df["F1 Score"].max(),4))

        
        
        st.subheader("Accuracy Comparison")
        accuracy_chart(results_df)

        st.subheader("Precision Comparison")
        precision_chart(results_df)

        st.subheader("Recall Comparison")
        recall_chart(results_df)

        st.subheader("F1 Score Comparison")
        f1_chart(results_df)

        st.subheader("Training Time Comparison")
        
        training_time_chart(results_df)

    else:

        st.info("⚠ Please train the models first.")


with tab8:

    st.header("⬇ Download Model")

    if "results_df" in st.session_state:

        results_df = st.session_state["results_df"]

        model_information(results_df)

        download_results(results_df)

        download_model()

    else:

        st.warning("Please train the model first.")


with tab9:

    st.header(" About Project")

    st.markdown("""
# Customer Churn Prediction Dashboard

This project predicts whether a bank customer is likely to leave the bank using Machine Learning algorithms.

### Features

- Dataset Analysis
- Data Preprocessing
- Model Training
- Model Comparison
- Customer Prediction
- Interactive Visualizations
- Download Best Model

### Machine Learning Algorithms

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Matplotlib
- Joblib

### Developed By

**Vishal Yadav**

B.Tech CSE (Artificial Intelligence)

Data Science & Machine Learning Enthusiast
""")

st.markdown("---")

st.caption(
    "© 2026 Vishal Yadav | Customer Churn Prediction Dashboard"
)