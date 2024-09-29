import os
import io
import json
import logging
import streamlit as st
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Dict
from PyPDF2 import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from googleapiclient.discovery import build
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langdetect import detect, LangDetectException
import google.generativeai as genai
import re
import difflib

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
genai_api_key = os.getenv("GENAI_API_KEY")
youtube_api_key = os.getenv("YOUTUBE_API_KEY")

# Configure GenAI
genai.configure(api_key=genai_api_key)

# Initialize session state variables
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

if "response_submitted" not in st.session_state:
    st.session_state.response_submitted = False

if "sources_shown" not in st.session_state:
    st.session_state.sources_shown = False

if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = {}

if "reference_texts_store" not in st.session_state:
    st.session_state.reference_texts_store = {}

if "document_store" not in st.session_state:
    st.session_state.document_store = []

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

    pdf_docs_with_names = read_files_from_folder(folder_path)
    if not pdf_docs_with_names or any(len(pdf) == 0 for pdf, _ in pdf_docs_with_names):
        raise HTTPException(status_code=400, detail="One or more PDF files are empty.")

    documents = get_all_pdfs_chunks(pdf_docs_with_names)
    pdf_vectorstore = get_vector_store(documents)

    st.session_state.vector_stores["pdf_vectorstore"] = pdf_vectorstore
    st.session_state.document_store.extend(documents)  # Store original documents

    st.success("PDFs processed successfully")

class QueryRequest(BaseModel):
    query: str

def get_response(context, question, model):
    chat_session = model.start_chat(history=[])

    prompt_template = """
    You are an AI assistant dedicated to answering questions based on the provided context.
    Answer the following question based on the reference context below.

    Context: {context}\n
    Question: {question}\n
    """

    try:
        response = chat_session.send_message(prompt_template.format(context=context, question=question))
        response_text = response.text

        if hasattr(response, 'safety_ratings') and response.safety_ratings:
            for rating in response.safety_ratings:
                if rating.probability != 'NEGLIGIBLE':
                    logging.warning("Response flagged due to safety concerns.")
                    return "", None, None

        logging.info(f"AI Response: {response_text}")
        return response_text
    except Exception as e:
        logging.warning(e)
        return ""

def generate_response(query_request: QueryRequest):
    if "pdf_vectorstore" not in st.session_state.vector_stores:
        st.error("PDFs must be processed first before generating a response.")
        return

    pdf_vectorstore = st.session_state.vector_stores['pdf_vectorstore']
    
    relevant_content = pdf_vectorstore.similarity_search(query_request.query, k=20)
    
    st.session_state.vector_stores["relevant_content"] = relevant_content

    context = " ".join([doc.page_content for doc in relevant_content])

    generation_config = {
        "temperature": 0.2,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 8000,
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction="You are a helpful document answering assistant."
    )
    
    response = get_response(context, query_request.query, model)
    st.session_state.vector_stores["response_text"] = response  # Store the response for later use
    return response

def generate_questions(context, num_questions, question_type, model):
    if question_type == "MCQ":
        prompt_template = f"""
        You are an AI assistant dedicated to the English language. Generate {num_questions} multiple-choice questions (MCQs) from the given context. 
        Create a set of MCQs with 4 answer options each. Ensure that the questions cover key concepts from the context provided and provide the correct answer as well. 
        Ensure the output is in JSON format with fields 'question', 'options', and 'correct_answer'.
        
        Context: {context}\n
        """
    else:
        prompt_template = f"""
        You are an AI assistant dedicated to the English language. Generate {num_questions} true/false questions from the given context. 
        For each true/false question, provide the correct answer as well. 
        Ensure the output is in JSON format with fields 'question' and 'correct_answer'.
        
        Context: {context}\n
        """

    try:
        response = model.start_chat(history=[]).send_message(prompt_template)
        response_text = response.text.strip()

        if response_text:
            response_json = clean_json_response(response_text)
            if response_json:
                return response_json
            else:
                return None
        else:
            return None
    except Exception as e:
        logging.warning(f"Error: {e}")
        return None

def clean_json_response(response_text):
    try:
        response_json = json.loads(response_text)
        return response_json
    except json.JSONDecodeError:
        try:
            cleaned_text = re.sub(r'```json', '', response_text).strip()
            cleaned_text = re.sub(r'```', '', cleaned_text).strip()

            match = re.search(r'(\{.*\}|\[.*\])', cleaned_text, re.DOTALL)
            if match:
                cleaned_text = match.group(0)
                response_json = json.loads(cleaned_text)
                return response_json
            else:
                logging.error("No JSON object or array found in response")
                return None
        except (ValueError, json.JSONDecodeError) as e:
            logging.error(f"Response is not a valid JSON: {str(e)}")
            return None

# Streamlit UI Components

st.title("AI Assistant for PDFs and YouTube Playlist Processing")

# Section to process PDFs and playlist
if st.button('🚀 Process PDFs and Playlist 🚀'):
    with st.spinner('Processing files...'):
       process_lessons_and_video()  # Call the function to process PDFs
    st.session_state.processing_complete = True  # Update session state

# Section for asking a query
if st.session_state.processing_complete:
    with st.form(key='response_form'):
        query = st.text_input("Ask a question related to the processed PDFs:")
        response_button = st.form_submit_button(label='Submit')

        if response_button:
            query_request = QueryRequest(query=query)
            response = generate_response(query_request)
            st.write("Response:", response)

# Section to generate questions
if st.session_state.processing_complete:
    with st.form(key='questions_form'):
        question_type = st.selectbox("Choose question type:", ["MCQ", "True/False"])
        questions_number = st.number_input("Number of questions:", min_value=1, max_value=10)
        generate_button = st.form_submit_button(label='Generate Questions')

        if generate_button:
            if "pdf_vectorstore" not in st.session_state.vector_stores:
                st.error("PDFs must be processed first before generating questions.")
            else:
                context = " ".join([doc.page_content for doc in st.session_state.vector_stores["relevant_content"]])
                model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    generation_config={"temperature": 0.2, "top_p": 1, "top_k": 1, "max_output_tokens": 8000}
                )
                questions = generate_questions(context, questions_number, question_type, model)
                st.write("Generated Questions:", questions)
