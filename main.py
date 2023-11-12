from dotenv import load_dotenv
import os
import streamlit as st 
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI


st.title("Prompt-driven data analysis with PandasAI")

load_dotenv()

API_KEY = os.environ['OPENAI_API_KEY']
#print(API_KEY)

# create an LLM by instantiating OpenAI object, and passing API token
llm = OpenAI(api_token=API_KEY)

# create PandasAI object, passing the LLM
pandas_ai = PandasAI(llm)

uploaded_csv_file = st.file_uploader("Upload a CSV file for analysis", type=['csv'])

if uploaded_csv_file is not None:
    df = pd.read_csv(uploaded_csv_file)
    st.write(df.head(3))

    prompt = st.text_area("Enter your prompt:")

    if st.button("Generate"):
        if prompt:
            #st.write("PandasAI is generating an answer, Please wait...")
            with st.spinner("Generating response..."):
                st.write(pandas_ai.run(df, prompt))
        else:
            st.warning("Please enter a prompt.")
