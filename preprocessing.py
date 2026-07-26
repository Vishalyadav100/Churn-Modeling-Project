import pandas as pd
from sklearn.preprocessing import (LabelEncoder, StandardScaler)
from sklearn.model_selection import train_test_split

def fill_missing_values(df):
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna("Unknown")
    return df


def remove_duplicates(df):
    df = df.drop_duplicates()
    return df


def encode_data(df):  # label encoding
    encoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == "object":df[col] = encoder.fit_transform(df[col])
    return df


def split_features_target(df,target): # feature and target
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def split_data(X,y,test_size,random_state): # train test and split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)
    return X_train, X_test, y_train, y_test


def scale_data(X_train,X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform( X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def preprocess_pipeline(df,target,test_size,random_state):

    # Fill Missing Values
    df = fill_missing_values(df)

    # Remove Duplicates
    df = remove_duplicates(df)

    df = df.drop(columns=["RowNumber","CustomerId","Surname"])

    # Encode Dataset
    df = encode_data(df)

    # Feature Target Split
    X, y = split_features_target(df,target)

    # Train Test Split
    X_train, X_test, y_train, y_test = split_data(X,y,test_size,random_state)

    # Scaling
    X_train, X_test = scale_data(X_train,X_test)

    return (df,X_train,X_test,y_train,y_test,X.columns)

    