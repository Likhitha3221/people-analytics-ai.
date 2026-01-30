import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent

# Standard Page Config
st.set_page_config(page_title="People Analytics AI", layout="wide")
st.title("📊 People Analytics AI & Visualization")

api_key = st.text_input("Paste your Groq API Key:", type="password")

if api_key:
    try:
        # Load the data
        df = pd.read_csv('people_analytics_data.csv')
        
        # Initialize Groq
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile", 
            groq_api_key=api_key,
            temperature=0
        )
        
        # Initialize the Agent
        agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=True, 
            allow_dangerous_code=True
        )
        
        query = st.text_input("Ask a question or request a chart (e.g., 'Show me a bar chart of...')")
        
        if query:
            with st.spinner("Generating Insights & Charts..."):
                # We tell the AI to use Matplotlib but NOT to use plt.show()
                # because Streamlit needs to handle the display.
                full_query = f"{query}. If a chart is requested, use matplotlib and ensure the plot is created."
                
                response = agent.run(full_query)
                st.write(response)
                
                # GRAPHS LOGIC: Check if the AI created a figure
                fig = plt.gcf()  # Get current figure
                if fig.get_axes():  # If there are axes, there is a chart
                    st.pyplot(fig)
                    plt.clf()  # Clear for the next question
                
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please enter your Groq API Key to begin.")