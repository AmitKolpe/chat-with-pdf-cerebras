# app_cerebras.py

import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_cerebras import ChatCerebras
from langchain_text_splitters import CharacterTextSplitter


from htmlTemplate import css, bot_template, user_template 
from dotenv import load_dotenv
load_dotenv()





 # custom HTML templates


# ---------------- PDF Text Extraction ----------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


# ---------------- Text Chunking ----------------
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# ---------------- Vectorstore ----------------
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# ---------------- Cerebras LLM Setup ----------------
def get_cerebras_llm():
    model = ChatCerebras(
        model="llama-4-scout-17b-16e-instruct",
        api_key=os.getenv("CEREBRAS_API_KEY")
    )
    return model


# ---------------- Conversation Chain ----------------
def get_conversation_chain(vectorstore):
    llm = get_cerebras_llm()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain


# ---------------- Handle User Input ----------------
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        msg_text = message.content if hasattr(message, 'content') else message
        if i % 2 == 0:
            st.markdown(user_template.replace("{{MSG}}", msg_text), unsafe_allow_html=True)
        else:
            st.markdown(bot_template.replace("{{MSG}}", msg_text), unsafe_allow_html=True)


# ---------------- Main Streamlit App ----------------
def main():
    st.set_page_config(page_title="Chat With Multiple PDFs (Cerebras)", page_icon="🧠")
    st.write(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat With Multiple PDFs 🧠 ")

    # Sidebar: PDF upload and processing
    with st.sidebar:
        st.subheader("📄 Upload PDFs")
        pdf_docs = st.file_uploader("Upload your PDFs and click 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("✅ Processing complete! You can now ask questions.")

    # User input after PDF processing
    if st.session_state.conversation:
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)


if __name__ == '__main__':
    main()
