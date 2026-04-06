import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# import openai
# from langchain_openai import ChatOpenAI
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM

# from langchain_groq import ChatGroq


import os
from dotenv import load_dotenv
load_dotenv()

## langsmith tracking
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_PROJECT']="Simple Q&A Chatbot with Grok"


## prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are an helpful assistant. Please respond to the user queries."),
        ("user","Question:{question}")
    ]
)

def generate_response(question,engine,temperature,max_tokens):
    # openai.api_key = api_key
    llm = OllamaLLM(model=engine)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({
        'question':question
    })
    return answer


## Title of the app
st.title("Enhanced Q&A chatbot with OPENAI")

# ## sidebar for setting
# st.sidebar.title("Settings")
# api_key = st.sidebar.text_input("Enter you AI api_key",type="password")

## drop down to select various llm model 
engine = st.sidebar.selectbox("Select a model",[
        "qwen3:4b",
        "gemma:2b"
    ])

## adjust response parameter
temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value = 0.7)
max_token = st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value = 150)

## Main interface 

st.write("Go ahead and ask any question")


user_input = st.text_input("You: ")

if user_input:
    response = generate_response(user_input,engine,temperature,max_token)
    st.write(response)
else:
    st.write("please provide the query")

