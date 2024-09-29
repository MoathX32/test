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
from langdetect import detect, LangDetectException
import google.generativeai as genai
import re
import difflib

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
genai_api_key = os.getenv("GENAI_API_KEY")

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

if "response_text" not in st.session_state:
    st.session_state.response_text = ""

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

def process_lessons():
    folder_path = "./Data"  # Automatically set to the "Data" folder in the current directory

    pdf_docs_with_names = read_files_from_folder(folder_path)
    if not pdf_docs_with_names or any(len(pdf) == 0 for pdf, _ in pdf_docs_with_names):
        raise HTTPException(status_code=400, detail="One or more PDF files are empty.")

    documents = get_all_pdfs_chunks(pdf_docs_with_names)
    pdf_vectorstore = get_vector_store(documents)

    st.session_state.vector_stores["pdf_vectorstore"] = pdf_vectorstore
    st.session_state.document_store.extend(documents)  # Store original documents

    st.success("تمت معالجة ملفات PDF بنجاح")

class QueryRequest(BaseModel):
    query: str

def get_response(context, question, model):
    chat_session = model.start_chat(history=[])

    # Updated prompt for response generation
    prompt_template = """
    أنت مساعد ذكي متخصص في اللغة العربية. أجب عن السؤال التالي بناءً على النص المرجعي المتاح.
    يمكنك توليد محتوى إضافي وتقديم أمثلة تدعم الإجابة، بشرط أن تبقى ضمن إطار الموضوع دون الخروج عنه.

    النص المرجعي: {context}\n
    السؤال: {question}\n
    """

    try:
        response = chat_session.send_message(prompt_template.format(context=context, question=question))
        response_text = response.text

        if hasattr(response, 'safety_ratings') and response.safety_ratings:
            for rating in response.safety_ratings:
                if rating.probability != 'NEGLIGIBLE':
                    logging.warning("تم الإبلاغ عن الإجابة لأسباب تتعلق بالسلامة.")
                    return "", None, None

        logging.info(f"إجابة الذكاء الاصطناعي: {response_text}")
        return response_text
    except Exception as e:
        logging.warning(e)
        return ""


def generate_response(query_request: QueryRequest):
    if "pdf_vectorstore" not in st.session_state.vector_stores:
        st.error("يجب معالجة ملفات PDF أولاً قبل توليد الإجابة.")
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
        system_instruction="أنت مساعد ذكي متخصص في تقديم إجابات دقيقة باللغة العربية."
    )
    
    response = get_response(context, query_request.query, model)
    st.session_state.response_text = response  # Store the response text for generating questions
    return response

def generate_questions_from_response(num_questions, question_type, model):
    # Use the AI-generated response text to generate questions
    response_text = st.session_state.response_text
    
    if not response_text:
        st.error("لم يتم توليد أي استجابة. يرجى توليد استجابة أولاً قبل توليد الأسئلة.")
        return None

    if question_type == "MCQ":
        # Updated prompt for intelligent question generation
        prompt_template = f"""
        أنت مساعد ذكي متخصص في اللغة العربية. قم بتوليد {num_questions} من أسئلة الاختيار من متعدد (MCQs) بناءً على الإجابة التالية.
        تأكد أن الأسئلة ذكية وتعتمد على التحليل، ويمكنك تقديم أمثلة لتوضيح المفاهيم. 
        يجب أن يحتوي كل سؤال على 4 خيارات وإجابة صحيحة، ويجب أن تبقى ضمن إطار الموضوع دون الخروج عنه.

        الإجابة: {response_text}\n
        """
    else:
        # Updated prompt for intelligent True/False question generation
        prompt_template = f"""
        أنت مساعد ذكي متخصص في اللغة العربية. قم بتوليد {num_questions} من أسئلة صح/خطأ بناءً على الإجابة التالية.
        تأكد أن الأسئلة ذكية وتساهم في اختبار فهم الطالب، ويمكنك تقديم أمثلة إذا لزم الأمر.
        يجب أن تحتوي الأسئلة على إجابات صحيحة، ويجب أن تبقى ضمن إطار الموضوع دون الخروج عنه.

        الإجابة: {response_text}\n
        """

    try:
        response = model.start_chat(history=[]).send_message(prompt_template)
        response_text = response.text.strip()

        # Log the response text to debug potential issues
        logging.info(f"AI Model Response for questions: {response_text}")

        if response_text:
            response_json = clean_json_response(response_text)
            if response_json:
                return response_json
            else:
                logging.error("الاستجابة لم تكن بصيغة JSON. تأكد من صحة الصيغة في النموذج.")
                st.error("لم يتم توليد الأسئلة بصيغة JSON. يرجى المحاولة مرة أخرى.")
                return None
        else:
            st.error("لم يتم استلام أي استجابة من النموذج.")
            return None
    except Exception as e:
        logging.warning(f"خطأ: {e}")
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
                logging.error("لم يتم العثور على كائن JSON في الاستجابة")
                return None
        except (ValueError, json.JSONDecodeError) as e:
            logging.error(f"الاستجابة ليست JSON صالحة: {str(e)}")
            return None

# Streamlit UI Components

st.title("مساعد ذكي لمعالجة ملفات PDF")

# Section to process PDFs
if st.button('🚀 معالجة ملفات PDF 🚀'):
    with st.spinner('جارٍ معالجة الملفات...'):
       process_lessons()  # Call the function to process PDFs
    st.session_state.processing_complete = True  # Update session state

# Section for asking a query
if st.session_state.processing_complete:
    with st.form(key='response_form'):
        query = st.text_input("اكتب سؤالك المتعلق بملفات PDF المعالجة:")
        response_button = st.form_submit_button(label='إرسال')

        if response_button:
            query_request = QueryRequest(query=query)
            response = generate_response(query_request)
            st.write("الإجابة:", response)

# Section to generate questions
if st.session_state.processing_complete:
    if st.session_state.response_text:
        with st.form(key='questions_form'):
            question_type = st.selectbox("اختر نوع السؤال:", ["MCQ", "صح/خطأ"])
            questions_number = st.number_input("عدد الأسئلة:", min_value=1, max_value=10)
            generate_button = st.form_submit_button(label='توليد الأسئلة')

            if generate_button:
                model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    generation_config={"temperature": 0.2, "top_p": 1, "top_k": 1, "max_output_tokens": 8000}
                )
                questions = generate_questions_from_response(questions_number, question_type, model)
                st.write("الأسئلة المتولدة:", questions)
