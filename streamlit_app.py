import os
import io
import logging
import streamlit as st
from fastapi import HTTPException
from pydantic import BaseModel
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
genai_api_key = os.getenv("GENAI_API_KEY")
genai.configure(api_key=genai_api_key)

# Initialize session state
if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = {}
if "document_store" not in st.session_state:
    st.session_state.document_store = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Store chat history in session state

# Function to get text chunks from a PDF file
def get_single_pdf_chunks(pdf_bytes, filename, text_splitter):
    pdf_stream = io.BytesIO(pdf_bytes)
    pdf_reader = PdfReader(pdf_stream)
    pdf_chunks = []
    for page_num, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        if page_text:
            page_chunks = text_splitter.split_text(page_text)
            for chunk in page_chunks:
                document = Document(page_content=chunk, metadata={"page": page_num, "filename": filename})
                pdf_chunks.append(document)
    return pdf_chunks

# Function to extract chunks from multiple PDFs
def get_all_pdfs_chunks(pdf_docs_with_names):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=990)
    all_chunks = []
    for pdf_bytes, filename in pdf_docs_with_names:
        pdf_chunks = get_single_pdf_chunks(pdf_bytes, filename, text_splitter)
        all_chunks.extend(pdf_chunks)
    return all_chunks

# Read PDF files from a folder
def read_files_from_folder(folder_path):
    pdf_docs_with_names = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            with open(os.path.join(folder_path, filename), "rb") as file:
                pdf_docs_with_names.append((file.read(), filename))
    return pdf_docs_with_names

# Get vector store from PDF documents
def get_vector_store(documents):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    return vectorstore

# Process PDFs and YouTube videos (stubbed YouTube playlist handling)
def process_lessons_and_video():
    folder_path = "./Data"
    pdf_docs_with_names = read_files_from_folder(folder_path)
    documents = get_all_pdfs_chunks(pdf_docs_with_names)
    pdf_vectorstore = get_vector_store(documents)
    st.session_state.vector_stores["pdf_vectorstore"] = pdf_vectorstore
    st.session_state.document_store.extend(documents)
    st.success("PDFs processed successfully")

# Data structure for queries
class QueryRequest(BaseModel):
    query: str

# Generate a response from Google Generative AI
def get_response(context, question, model):
    chat_history = st.session_state.chat_history
    
    # Format the chat history for Google Generative AI
    formatted_history = [{"role": message["role"], "parts": [{"text": message["content"]}]} for message in chat_history]
    
    # Start a new chat session with formatted history
    chat_session = model.start_chat(history=formatted_history)
    
    # Define the response template
    prompt_template = f"""
    أجب على السؤال التالي بناءً على الدروس المرجعية.
    السياق: {context}\n
    السؤال: {question}\n
    """
    
    # Send the message and receive the response
    response = chat_session.send_message(prompt_template)
    response_text = response.text
    
    # Append to chat history
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "model", "content": response_text})
    
    # Update session state with chat history
    st.session_state.chat_history = chat_history
    
    return response_text

# Generate the response for a user query
def generate_response(query_request: QueryRequest):
    pdf_vectorstore = st.session_state.vector_stores.get('pdf_vectorstore')
    if not pdf_vectorstore:
        st.error("PDFs must be processed first.")
        return
    
    # Search the vectorstore for relevant content
    relevant_content = pdf_vectorstore.similarity_search(query_request.query, k=20)
    
    # Combine relevant content into context
    context = " ".join([doc.page_content for doc in relevant_content])
    
    # Initialize the model for response generation
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config={"temperature": 0.2, "top_p": 1, "max_output_tokens": 8000}
    )
    
    # Generate response based on context and query
    response = get_response(context, query_request.query, model)
    
    return response

# Streamlit UI Components
st.title("مساعد اللغة العربية")

# Button to start processing PDFs and YouTube videos
if st.button('ابدأ تشغيل المساعد'):
    process_lessons_and_video()

# Form to submit queries
with st.form(key='response_form'):
    query = st.text_input("كيف يمكنني مساعدتك:")
    response_button = st.form_submit_button(label='أجب')
    
    # Generate response when the button is clicked
    if response_button:
        query_request = QueryRequest(query=query)
        response = generate_response(query_request)
        st.write("الرد:", response)
