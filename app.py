import streamlit as st
import pandas as pd
import os
from ydata_profiling import ProfileReport

from pycaret.regression import setup, compare_models, pull, save_model

with st.sidebar:
    st.title("ML Data analyzer")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
    st.info("This is an automated ML pipeline")

df = None  # Initialize df outside the if conditions to avoid NameError

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
    st.title("Upload your data for modeling")
    file = st.file_uploader("Upload your dataset here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Data analysis & Visualizer")
    if df is not None:
        with st.spinner("Profiling the data... This may take a moment."):
            profile = ProfileReport(df)

        # Generate the HTML report
        profile_html = profile.to_html()

        # Display the HTML report in Streamlit
        st.components.v1.html(profile_html, height=800)

if choice == "ML":
    st.title("Machine Learning")
    target = st.selectbox("Select Target", df.columns)

    # Convert the target variable to a numeric type
    if df[target].dtype == 'O':  # Check if target variable is of object (string) type
        df[target] = df[target].astype('category').cat.codes

    setup(df, target=target, verbose=False)
    setup_df = pull()
    st.info("Experimental Settings")
    st.dataframe(setup_df)
    best_model = compare_models()
    compare_df = pull()
    st.info("ML Model")
    st.dataframe(compare_df)

if choice == "Download":
    pass
