import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent,AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()


## Arxiv and wikipedia Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200) # this is a wrapper which decides what to bring in content after scraping the arxiv paper for the input 
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)
# ArxivQueryRun it is a tool through which our llm interact and bring output.

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

search=DuckDuckGoSearchRun(name="Search")


st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API Key:",type="password")


# my entire coversation have chat_history 
if "messages" not in st.session_state:
    st.session_state["messages"]=[{
        "role":"assistant",
        "content":"Hi,I'm a chatbot who can search the web. How can I help you?"
    }]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])


if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=api_key,model_name="openai/gpt-oss-120b",streaming=True)# streaming = true means Enables token-by-token output
    tools=[search,arxiv,wiki]


    # Creates an agent pipeline
    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True,verbose=True) # verbose = true -> result-> Thought → Action → Observation → Final Answer (sab print karta hai internally)


    # '''
    #     ZERO_SHOT_REACT_DESCRIPTION

    #     Break it:
    #     ZERO_SHOT → no training examples
    #     REACT → Reason + Act loop
    #     DESCRIPTION → uses tool descriptions
    # '''
    # '''
    # handle_parsing_errors=True -> prevents crash retries automatically
    # '''
    # '''
    #     User Question →
    #         LLM thinks →
    #         "Should I use a tool?" →
    #         Chooses tool →
    #         Executes tool →
    #         Gets result →
    #         Final answer
    # '''

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=True)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)
