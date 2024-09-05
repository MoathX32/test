import os
import io
import json
import logging
import streamlit as st
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Dict
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
import google.generativeai as genai
import re

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
genai_api_key = os.getenv("GENAI_API_KEY")

# Configure GenAI
genai.configure(api_key=genai_api_key)

# Initialize global stores
if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = {}
if "reference_texts_store" not in st.session_state:
    st.session_state.reference_texts_store = {}
if "document_store" not in st.session_state:
    st.session_state.document_store = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Store the chat history

# Function Definitions
def get_single_pdf_chunks(pdf_bytes, filename, text_splitter):
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty PDF content.")
        
    pdf_stream = io.BytesIO(pdf_bytes)
    pdf_reader = PdfReader(pdf_stream)
    pdf_chunks = []
    for page_num, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        if page_text:
            page_chunks = text_splitter.split_text(page_text)
            for chunk in page_chunks:
                document = Document(page_content=chunk, metadata={"page": page_num, "filename": filename})
                logging.info(f"Adding document chunk with metadata: {document.metadata}")
                pdf_chunks.append(document)
    return pdf_chunks

def get_all_pdfs_chunks(pdf_docs_with_names):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=990
    )

    all_chunks = []
    for pdf_bytes, filename in pdf_docs_with_names:
        pdf_chunks = get_single_pdf_chunks(pdf_bytes, filename, text_splitter)
        all_chunks.extend(pdf_chunks)
    return all_chunks

def read_files_from_folder(folder_path):
    pdf_docs_with_names = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            with open(os.path.join(folder_path, filename), "rb") as file:
                pdf_docs_with_names.append((file.read(), filename))
    return pdf_docs_with_names

def get_vector_store(documents):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
        return vectorstore
    except Exception as e:
        logging.warning("Issue with creating the vector store.")
        raise HTTPException(status_code=500, detail="Issue with creating the vector store.")

def process_lessons_and_video():
    folder_path = "./Data"  # Automatically set to the "Data" folder in the current directory
    playlist_url = "https://www.youtube.com/watch?v=DFyPl2cZM2g&list=PLX1bW_GeBRhDkTf_jbdvBbkHs2LCWVeXZ"  # Always set to the provided URL

    pdf_docs_with_names = read_files_from_folder(folder_path)
    if not pdf_docs_with_names or any(len(pdf) == 0 for pdf, _ in pdf_docs_with_names):
        raise HTTPException(status_code=400, detail="One or more PDF files are empty.")

    documents = get_all_pdfs_chunks(pdf_docs_with_names)
    pdf_vectorstore = get_vector_store(documents)

    playlist_id = playlist_url.split("list=")[-1].split("&")[0]

    st.session_state.vector_stores["pdf_vectorstore"] = pdf_vectorstore
    st.session_state.vector_stores["playlist_id"] = playlist_id
    st.session_state.document_store.extend(documents)  # Store original documents

    st.success("PDFs and playlist processed successfully")

class QueryRequest(BaseModel):
    query: str

def get_response(context, question, model):
    # Retrieve the existing chat history from session state
    chat_history = st.session_state.chat_history
    
    # Start a new chat session, passing in the chat history
    chat_session = model.start_chat(history=chat_history)

    # Define the prompt template
    prompt_template = """
    أنت مساعد ذكي في مادة اللغة العربية للصفوف الأولى. تفهم أساسيات اللغة العربية مثل الحروف، الكلمات البسيطة، والجمل الأساسية.
    لاتجب الا اذا كان السؤال واضح او استفهم من الطالب المقصود
    أجب على السؤال التالي من خلال فهمك للدروس الموجوده في السياق المرجعي . قدم إجابة بسيطة وواضحة تتناسب مع مستوى الصفوف الأولى.
    يجب أن يكون الرد مفهومًا.
    لا تجب على أي سؤال خارج فهمك سياق النص.
    يمكنك ان تكون امثله وتعريفات لايشترط ان تكون مذكوره في النص
    السياق: {context}\n
    السؤال: {question}\n
    """

    try:
        # Send the question and context to the model
        response = chat_session.send_message(prompt_template.format(context=context, question=question))
        response_text = response.text

        # Check for safety ratings
        if hasattr(response, 'safety_ratings') and response.safety_ratings:
            for rating in response.safety_ratings:
                if rating.probability != 'NEGLIGIBLE':
                    logging.warning("Response flagged due to safety concerns.")
                    return "", None, None

        # Log and display the response
        logging.info(f"AI Response: {response_text}")

        # Append the question and response to the chat history
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": response_text})

        # Store the updated chat history in the session state
        st.session_state.chat_history = chat_history

        return response_text

    except Exception as e:
        logging.warning(e)
        return ""

# Display the chat history to the user
def display_chat_history():
    if "chat_history" in st.session_state and st.session_state.chat_history:
        st.write("### Conversation History")
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.write(f"**You**: {message['content']}")
            elif message["role"] == "assistant":
                st.write(f"**Assistant**: {message['content']}")
    else:
        st.write("No conversation history available.")

# Generate response UI
def generate_response(query_request: QueryRequest):
    if "pdf_vectorstore" not in st.session_state.vector_stores:
        st.error("PDFs must be processed first before generating a response.")
        return

    pdf_vectorstore = st.session_state.vector_stores['pdf_vectorstore']
    
    # Search for relevant content from the PDFs
    relevant_content = pdf_vectorstore.similarity_search(query_request.query, k=20)
    
    st.session_state.vector_stores["relevant_content"] = relevant_content

    # Combine the relevant content for context
    context = " ".join([doc.page_content for doc in relevant_content])

    # Set up the model generation config
    generation_config = {
        "temperature": 0.2,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 8000,
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config=generation_config,
        system_instruction="You are a helpful document answering assistant."
    )
    
    # Get the response from the model based on the context and the user's query
    response = get_response(context, query_request.query, model)
    st.session_state.vector_stores["response_text"] = response  # Store the response for later use

    # Display the chat history
    display_chat_history()

    return response

# Inside Streamlit form (UI for query input and response generation)
if st.session_state.processing_complete:
    with st.form(key='response_form'):
        query = st.text_input("كيف يمكنني مساعدتك:")
        response_button = st.form_submit_button(label='أجب')

        if response_button:
            query_request = QueryRequest(query=query)
            response = generate_response(query_request)
            st.write("الرد:", response)

    # Display the chat history even after response submission
    display_chat_history()
