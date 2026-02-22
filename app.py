import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# LangChain - correct modern imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage

# Cerebras
from langchain_cerebras import ChatCerebras

from htmlTemplate import css, bot_template, user_template

load_dotenv()


# -------------------------------------------------
# PDF TEXT EXTRACTION
# -------------------------------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


# -------------------------------------------------
# TEXT SPLITTING
# -------------------------------------------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


# -------------------------------------------------
# VECTOR STORE
# -------------------------------------------------
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(text_chunks, embedding=embeddings)


# -------------------------------------------------
# CEREBRAS LLM
# -------------------------------------------------
def get_cerebras_llm():
    return ChatCerebras(
        model="llama3.1-8b",
        api_key=os.getenv("CEREBRAS_API_KEY"),
        temperature=0.2
    )


# -------------------------------------------------
# HANDLE USER INPUT
# No langchain.chains used - built manually to avoid import issues
# -------------------------------------------------
def handle_userinput(user_question):
    llm = get_cerebras_llm()
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

    # Get relevant docs from vector store
    docs = retriever.invoke(user_question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build chat history string for context
    history_text = ""
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            history_text += f"Human: {msg.content}\n"
        else:
            history_text += f"Assistant: {msg.content}\n"

    # Build full prompt
    full_prompt = f"""You are a helpful assistant. Use the context below to answer the question.

Context from documents:
{context}

Chat History:
{history_text}
Human: {user_question}
Assistant:"""

    # Call LLM
    response = llm.invoke(full_prompt)
    answer = response.content

    # Save to chat history
    st.session_state.chat_history.append(HumanMessage(content=user_question))
    st.session_state.chat_history.append(AIMessage(content=answer))

    # Display full chat history
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.markdown(
                user_template.replace("{{MSG}}", msg.content),
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                bot_template.replace("{{MSG}}", msg.content),
                unsafe_allow_html=True
            )


# -------------------------------------------------
# STREAMLIT APP
# -------------------------------------------------
def main():
    st.set_page_config(
        page_title="Chat with Multiple PDFs (Cerebras)",
        page_icon="🧠"
    )

    st.write(css, unsafe_allow_html=True)

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    st.header("Chat with Multiple PDFs 🧠")

    # Sidebar for PDF upload
    with st.sidebar:
        st.subheader("📄 Upload PDFs")
        pdf_docs = st.file_uploader(
            "Upload your PDFs and click Process",
            accept_multiple_files=True,
            type="pdf"
        )

        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("Could not extract text from the uploaded PDFs.")
                        return
                    chunks = get_text_chunks(raw_text)
                    st.session_state.vectorstore = get_vectorstore(chunks)
                    st.session_state.chat_history = []  # Reset history on new upload
                    st.session_state.conversation = True  # Mark as ready

                st.success("✅ PDFs processed successfully!")

    # Chat interface
    if st.session_state.conversation:
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)
    else:
        st.info("👈 Please upload and process your PDFs from the sidebar to start chatting.")


if __name__ == "__main__":
    main()

#venv\Scripts\python.exe -m streamlit run app.py