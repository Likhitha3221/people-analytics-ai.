import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent

st.set_page_config(page_title="People Analytics AI", layout="wide")
st.title("📊 People Analytics AI & Visualization")

api_key = st.text_input("Paste your Groq API Key:", type="password")

if api_key:
    try:
        df = pd.read_csv('people_analytics_data.csv')
        
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile", 
            groq_api_key=api_key,
            temperature=0
        )
        
        # FIX 1: Added handle_parsing_errors=True
        agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=True, 
            allow_dangerous_code=True,
            handle_parsing_errors=True 
        )
        
        query = st.text_input("Ask a question or request a chart:")
        
        if query:
            with st.spinner("Processing..."):
                # FIX 2: Better system instructions to prevent the AI from "explaining" the code
                full_query = (
                    f"{query}. If a chart is requested, write and execute the python code "
                    "using matplotlib. Do not explain the code, just execute it. "
                    "End your response with 'Final Answer: [your summary]'."
                )
                
                response = agent.run(full_query)
                st.write(response)
                
                # Show the plot if one was created
                fig = plt.gcf()
                if fig.get_axes():
                    st.pyplot(fig)
                    plt.clf()
                
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please enter your Groq API Key to begin.")
