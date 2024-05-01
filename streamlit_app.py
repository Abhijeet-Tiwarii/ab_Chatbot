# Import necessary libraries
import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
# from flask import Flask, render_template, request, redirect
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import re
#
OPENAI_API_KEY = 'Your key'


os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

vectorstore = None
conversation_chain = None
chat_history = []

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=10,
        chunk_overlap=2,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


# Streamlit
# streamlit run pdf_app2.py
def main():
    global text, chunks, vectorstore, conversation_chain, chat_history

    st.title('My UCLan Boat')

    # pdf_docs = st.file_uploader("Upload PDF", type=['pdf'], accept_multiple_files=True)

    with open(r"C:\Users\abhijeet.a.tiwari\basic_chat\pdfGPT-main\upload_data_folder\About_UCLan.pdf", "r", encoding="latin-1") as file:
        raw_text = file.read()

    # if pdf_docs is not None:
        # raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        conversation_chain = get_conversation_chain(vectorstore)

        user_question = st.text_input("Enter your question:")
        if st.button('Submit'):
            response = conversation_chain({'question': user_question})
            chat_history = response['chat_history']
            # st.write(chat_history[1])

            print("Type of it", type(chat_history[1]))
            print("Dir of it", dir(chat_history[1]))



            chat_history[1].content = chat_history[1].content.replace('content=', '')


            links = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                               chat_history[1].content)
            for link in links:
                chat_history[1].content = chat_history[1].content.replace(link, f'{link}')


            chat_history[1].content = chat_history[1].content.replace('\n', '<br/>')


            st.markdown(chat_history[1].content, unsafe_allow_html=True)

if __name__ == '__main__':
    main()

