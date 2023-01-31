import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load dataset iris.csv
df = pd.read_csv('iris.csv')

# Create random forest classifier model
clf = RandomForestClassifier()
clf.fit(df.drop(['Species'], axis=1), df['Species'])

# Create UI
st.title("Iris Flower Prediction")
sepal_length = st.number_input("Sepal Length", min_value=0, max_value=10)
sepal_width = st.number_input("Sepal Width", min_value=0, max_value=10)
petal_length = st.number_input("Petal Length", min_value=0, max_value=10)
petal_width = st.number_input("Petal Width", min_value=0, max_value=10)

if st.button("Submit"):
    result = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    st.success(f"The predicted iris flower type is: {result[0]}")
