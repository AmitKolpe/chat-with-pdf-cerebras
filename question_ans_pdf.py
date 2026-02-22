import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()

st.title("PDF Question Answering Bot")

# FIX: type must be lowercase 'pdf'
file_upload = st.file_uploader("Upload a PDF file", type="pdf")

if file_upload:
    # FIX: PyPDFLoader requires a file path, not an UploadedFile object.
    # Save the uploaded file to a temporary file first.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_upload.read())
        tmp_path = tmp_file.name

    with st.spinner("Loading and processing PDF..."):
        # Load PDF from temp file path
        pdf_loader = PyPDFLoader(tmp_path)
        document = pdf_loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks = splitter.split_documents(document)

        # FIX: HuggingFaceEmbeddings uses 'model_name', not 'repo_id'
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Build vector store
        vectorstore = FAISS.from_documents(text_chunks, embeddings)

        # LLM via HuggingFace Endpoint
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # reliable free model
            task="text-generation",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            max_new_tokens=512,
            temperature=0.2
        )

        # Build RetrievalQA chain
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
        )

    st.success("✅ PDF processed! Ask your question below.")

    question = st.text_input("Ask a question about the PDF:")

    if question:
        with st.spinner("Finding answer..."):
            # FIX: use .invoke() instead of deprecated .run()
            result = chain.invoke({"query": question})
            answer = result.get("result", "No answer found.")
        st.markdown("### Answer")
        st.write(answer)

    # Clean up temp file
    os.unlink(tmp_path)
