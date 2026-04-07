from fastapi import FastAPI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes

import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=groq_api_key,
    
)

# create prompt template
system_template = "Translate the following in the {language}: "

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",system_template),
        ("user","{text}")
    ]
)

parser = StrOutputParser()


## create chain
chain = prompt_template | model | parser

## app defination

app = FastAPI(title="Langchain Server", version="1.0",description="A simple API server using using langchain runnable interfaces")

#adding chain routes

add_routes(
    app,
    chain,
    path = "/chain"
)

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)
