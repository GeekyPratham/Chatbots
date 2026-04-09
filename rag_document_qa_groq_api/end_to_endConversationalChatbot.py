import streamlit as st
from langchain_classic.chains import create_retrieval_chain,create_history_aware_retriever
# Ye aisa retriever banata hai jo current question ke saath chat history bhi samajhta hai

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
# from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
# Prompt me dynamic jagah banane ke liye MessagesPlaceholder use hota hai
# Isse hum chat_history inject kar paate hain

from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory 
from langchain_core.chat_history import BaseChatMessageHistory


from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter # intelligent way to splitting the text so that it does not loose information
from langchain_community.document_loaders import PyPDFLoader

import os
import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

from dotenv import load_dotenv
load_dotenv()


# model which i will be using  for embedding which is free and porvided by Huggingface
embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2") 


## setup streamlit app
st.title("Conversation rag with pdf upload and chat history")

# user have to upload document or pdf in which they want to chat or query
st.write("Upload pdf and chat with their content") 

# Input the Groq api key means user have to put their own groq api key
api_key = st.text_input("Enter your Groq api key:" , type="password")

# check if groq api key is provided
if api_key and api_key.strip() != "":
    llm = ChatGroq(
        groq_api_key = api_key,
        model_name="openai/gpt-oss-20b",
        
    )
    
    # chat interface
    # creating session
    session_id = st.text_input("Session ID",value="default_session")

    ## state fully manage the chat history

    if 'store' not in st.session_state:
        st.session_state.store={}
    
    uploaded_files = st.file_uploader("Choose A pdf file",type="pdf",accept_multiple_files=True)

    ## process the uploaded files

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            # temppdf = f"./temp.pdf" # same file overwritten and last file only process
            temppdf = f"./temp_{uploaded_file.name}"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader = PyPDFLoader(temppdf) #Ye LangChain ka loader hai jo: PDF ko read karega ,usko structured format me convert karega
            docs=loader.load() # PDF ko pages me tod deta hai aur Har page ko ek document object banata hai
            documents.extend(docs) # Ye docs list ko main documents list me add kar deta hai

        
        # splites and creating embedding for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## splits the content so that it does not loose context 
        splits = text_splitter.split_documents(documents)
        vectorStore = Chroma.from_documents(documents=splits,embedding=embedding)# convert text into numerical and store it into vectorstore which is database
        
        retriever = vectorStore.as_retriever()


        ## prompt which act as a system message

        # Ye ek system prompt hai jo LLM ko instruction de raha hai
        # Iska kaam: agar user ka current question previous conversation pe depend karta hai,
        # toh us question ko standalone form me rewrite kar do

        contextualize_q_system_prompt=(
            ## ye first system prompt hai isse user ke question ke according related chunks retrive kar sake isme ye dhyan rake ki mere pas previous message history ho

            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        # Ab hum ek prompt template bana rahe hain jo LLM ko diya jayega
        # Is prompt me 3 cheezein hongi:
        # 1. system instruction
        # 2. purani chat history
        # 3. current user input

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )

        # Ye sabse important line hai
        # create_history_aware_retriever ek aisa retriever banata hai jo directly user ke question ko docs me search nahi karta
        # Balki:
        # Step 1: chat history + current question ko dekh kar standalone question banata hai
        # Step 2: us standalone question ko retriever me bhejta hai

        history_aware_retriever = create_history_aware_retriever(
            llm, # LLM jo question ko rewrite karega
            retriever, # actual retriever jo vector DB / docs me search karega
            contextualize_q_prompt # prompt jo LLM ko batayega rewrite kaise karna hai
            
        ) 

        ## Answer question prompt
        system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
        
        # Ab hum final answering ke liye ek QA prompt bana rahe hain
        # Is prompt ka kaam hai:
        # retrieved documents + chat history + user question ko use karke final answer dena
        qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        

        # Ye chain retrieved documents ko "stuff" karegi prompt me
        # "stuff" ka matlab:
        # jitne docs retrieve hue unko ek saath prompt me chipka do
        # aur phir LLM se answer generate karao

        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)


        # Ye full RAG pipeline ban rahi hai
        # Step 1: history_aware_retriever relevant docs nikaalega
        # Step 2: question_answer_chain un docs ke basis pe answer dega
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)


        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                # st.session_state.store ek dictionary hai
                # jisme har session ka chat history store hota hai
                # st.session_state.store = {
                #     "session1": ChatMessageHistory(),
                #     "session2": ChatMessageHistory()
                # }

                # if not present in dictionary Ek naya empty chat history object create karo
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session] # ush session ka chat history return karo
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history=get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                },  # constructs a key "abc123" in `store`.
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)

else:
    st.warning("Please enter the GRoq API Key")

