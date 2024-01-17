import streamlit as st
import pandas as pd

with st.sidebar:
    st.title("ML Data analyzer")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
    st.info("This is a automated ML pipeline")

if choice == "Upload":
    st.title("Upload your data for modeling")
    file = st.file_uploader("Upload your dataset here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)
       


if choice == "Profiling":
    pass

if choice == "ML":
    pass

if choice == "Download":
    pass

