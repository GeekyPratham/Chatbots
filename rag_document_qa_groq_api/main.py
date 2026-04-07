import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_classic.chains import create_retrieval_chain


from dotenv import load_dotenv
load_dotenv()

# load the groq api key
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name = "llama-3.3-70b-versatile"
)


prompt = ChatPromptTemplate.from_template(
    """ 
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question:{input}
    """
)

# st.session_state-> Streamlit har interaction pe script ko rerun karta hai, isliye data lose na ho uske liye st.session_state use karte hain. Ye current session ki temporary memory hoti hai jisme hum variables jaise vector store, embeddings, chat history, user input etc. store kar sakte hain.
def create_vector_embedding():
    if "vectors" not in st.session_state:
        # Free HuggingFace embeddings
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        st.session_state.loader=PyPDFDirectoryLoader("research_paper") # data ingestion step 
        st.session_state.docs = st.session_state.loader.load() # load the data from pdf and form document
        # processing / or embedding this whole data at once not possible due to context loose so we divide it into different chunks
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)## convert the documents into vectore using ollama embedding technique

user_prompt = st.text_input("Enter you query from the research paper")

if st.button('Document Embedding'):
    create_vector_embedding()
    st.write("Vector db is ready Now you can query")


import time

# if user_prompt:
#     document_chain = create_stuff_documents_chain(llm,prompt)
#     retriever = st.session_state.vectors.as_retriever()
#     retriever_chain =create_retrieval_chain(retriever,document_chain)

#     start=time.process_time()
#     response = retriever_chain.invoke({
#         'input':user_prompt
#     })
#     print(f"Response time:{time.process_time()-start}")
    
#     st.write(response['answer'])
#     print("other responses")

#     ## with a streamlit expander
#     with st.expander("Document similarity search"):
#         for i,doc in enumerate(response['context']):
#             st.write(doc.page_content)
#             st.write('-------------')


if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please click on 'Document Embedding' first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retriever_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retriever_chain.invoke({
            'input': user_prompt
        })
        print(f"Response time:{time.process_time()-start}")

        st.write(response['answer'])

        with st.expander("Document similarity search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('-------------')
