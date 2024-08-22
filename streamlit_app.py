import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
import google.generativeai as genai
import logging
import io
import re
import json

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
genai_api_key = os.getenv("GENAI_API_KEY")

# Configure GenAI
genai.configure(api_key=genai_api_key)

# Initialize session state if not already done
if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = {}
if "reference_texts_store" not in st.session_state:
    st.session_state.reference_texts_store = {}
if "document_store" not in st.session_state:
    st.session_state.document_store = []

# Constants
FOLDER_PATH = "Data"  # Fixed folder path
PLAYLIST_URL = "https://www.youtube.com/watch?v=DFyPl2cZM2g&list=PLX1bW_GeBRhDkTf_jbdvBbkHs2LCWVeXZ"  # Fixed YouTube playlist URL

# Function to process PDFs
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
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    return vectorstore

# Function to generate responses
# Function to generate responses with both study assistance and general chat capabilities
# Function to generate responses with both study assistance and general chat capabilities
# Function to generate responses with both study assistance and general chat capabilities
def get_response(context, question, model):
    # Initialize chat history if not already present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Ensure chat history entries are correctly formatted
    formatted_history = []
    for entry in st.session_state.chat_history:
        if "question" in entry and "response" in entry:
            formatted_history.append({
                "author": "user",
                "content": entry["question"]
            })
            formatted_history.append({
                "author": "bot",
                "content": entry["response"]
            })

    # Check if the question is related to the study content or a general chat
    is_study_related = any(keyword in question.lower() for keyword in ["درس", "قواعد", "سؤال", "معلومة", "شرح", "كتاب", "نص"])

    if is_study_related:
        # Study-related response using the provided context
        chat_session = model.start_chat(history=formatted_history)
        prompt_template = """
        أنت مساعد ذكي في مادة اللغة العربية للصفوف الأولى. مهمتك هي مساعدة الطلاب على فهم الدروس والإجابة على أسئلتهم باستخدام المعلومات الموجودة في الدروس فقط.
        استخدم النص الموجود في السياق المرجعي أدناه للإجابة على السؤال. إذا لم تتمكن من العثور على إجابة في النص، أخبر المستخدم أنك غير قادر على الإجابة بناءً على المعلومات المتاحة.
        السياق: {context}\n
        السؤال: {question}\n
        """
    else:
        # General chat response
        chat_session = model.start_chat(history=formatted_history)
        prompt_template = """
        أنت مساعد دردشة ذكي. يمكنك الدردشة مع الطالب والإجابة على أي أسئلة عامة أو بدء محادثة ودية.
        السؤال: {question}\n
        """

    try:
        response = chat_session.send_message(prompt_template.format(context=context if is_study_related else "", question=question))
        response_text = response.text.strip()

        # Save the current conversation in the chat history
        st.session_state.chat_history.append({"question": question, "response": response_text})

        if is_study_related and ("لا أستطيع الإجابة" in response_text or not response_text):
            return "أنا لا أستطيع الإجابة على هذا السؤال بناءً على المعلومات المتاحة في الدروس."
        else:
            return response_text
    except Exception as e:
        return "حدث خطأ أثناء محاولة الإجابة على سؤالك. من فضلك حاول مرة أخرى لاحقًا."

def extract_reference_texts_as_json(response_text, context):
    ref_prompt = f"""
    بناءً على الإجابة التالية، حدد النص الأكثر ارتباطًا من المستندات المرجعية الخاصة بمادة اللغة العربية، والتي يجب أن تتضمن عناوين دروس مثل "درس: قواعد اللغة العربية" أو "درس: أدب العصر العباسي".
    قدم العنوان الرئيسي للدرس كمفتاح 'filename'، وأضف النص الأكثر ارتباطًا فقط تحت مفتاح 'relevant_texts'.
    الإجابة: {response_text}
    السياق المرجعي: {context}
    """
    chat_session = genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config={"temperature": 0.2, "top_p": 1, "top_k": 1, "max_output_tokens": 8000}
    ).start_chat(history=[])
    ref_response = chat_session.send_message(ref_prompt)
    return clean_json_response(ref_response.text.strip())

# Function to generate questions
def generate_questions(relevant_text, num_questions, question_type, model):
    if question_type == "MCQ":
        prompt_template = f"""
        You are an AI assistant tasked with generating {num_questions} multiple-choice questions (MCQs) from the given context. \
        Create a set of MCQs with 4 answer options each. Ensure that the questions cover key concepts from the context provided and provide the correct answer as well. \
        Ensure the output is in JSON format with fields 'question', 'options', and 'correct_answer', and ensure the output language as context language.
        
        Context: {relevant_text}\n
        """
    else:
        prompt_template = f"""
        You are an AI assistant tasked with generating {num_questions} true/false questions from the given context. \
        For each true/false question, provide the correct answer as well. \
        Ensure the output is in JSON format with fields 'question' and 'correct_answer', and ensure the output language as context language.
        
        Context: {relevant_text}\n
        """

    response = model.start_chat(history=[]).send_message(prompt_template)
    response_text = response.text.strip()

    if response_text:
        return clean_json_response(response_text)
    return None

# Function to get playlist videos (mock implementation)
def get_playlist_videos(playlist_id):
    # This should be replaced with actual code that interacts with YouTube API to fetch playlist videos
    return [
        {"title": "حقوقي وواجباتي في البيت", "video_id": "abc123"},
        {"title": "كيف تنتخب مجلس القسم؟", "video_id": "def456"},
        {"title": "كيف نمارس مواطنتنا في المدرسة؟", "video_id": "ghi789"}
    ]

# Streamlit App
st.title("PDF and Video Processing App")

# Sidebar Buttons
st.sidebar.title("Navigation")
if st.sidebar.button("Process PDFs and Videos"):
    st.session_state.current_page = "Process PDFs and Videos"
if st.sidebar.button("Generate Response"):
    st.session_state.current_page = "Generate Response"
if st.sidebar.button("Generate Reference Texts"):
    st.session_state.current_page = "Generate Reference Texts"
if st.sidebar.button("Generate Video Segment URLs"):
    st.session_state.current_page = "Generate Video Segment URLs"
if st.sidebar.button("Generate Questions"):
    st.session_state.current_page = "Generate Questions"

# Default page
if "current_page" not in st.session_state:
    st.session_state.current_page = "Process PDFs and Videos"

# Page Routing
if st.session_state.current_page == "Process PDFs and Videos":
    st.header("Process PDFs and Videos")
    if st.button("Process"):
        pdf_docs_with_names = read_files_from_folder(FOLDER_PATH)
        documents = get_all_pdfs_chunks(pdf_docs_with_names)
        pdf_vectorstore = get_vector_store(documents)
        playlist_id = PLAYLIST_URL.split("list=")[-1].split("&")[0]
        st.session_state.vector_stores["pdf_vectorstore"] = pdf_vectorstore
        st.session_state.vector_stores["playlist_id"] = playlist_id
        st.session_state.document_store.extend(documents)
        st.success("PDFs and playlist processed successfully!")

elif st.session_state.current_page == "Generate Response":
    st.header("Generate Response")
    query = st.text_input("Enter your query:")
    if st.button("Generate"):
        if "pdf_vectorstore" not in st.session_state.vector_stores:
            st.error("PDFs must be processed first.")
        else:
            pdf_vectorstore = st.session_state.vector_stores['pdf_vectorstore']
            relevant_content = pdf_vectorstore.similarity_search(query, k=20)
            st.session_state.vector_stores["relevant_content"] = relevant_content
            context = " ".join([doc.page_content for doc in relevant_content])
            model = genai.GenerativeModel(
                model_name="gemini-1.5-pro-latest",
                generation_config={"temperature": 0.2, "top_p": 1, "top_k": 1, "max_output_tokens": 8000},
                system_instruction="You are a helpful document answering assistant."
            )
            response = get_response(context, query, model)
            st.session_state.vector_stores["response_text"] = response
            st.write(response)

elif st.session_state.current_page == "Generate Reference Texts":
    st.header("Generate Reference Texts")
    if st.button("Generate"):
        if "pdf_vectorstore" not in st.session_state.vector_stores or "response_text" not in st.session_state.vector_stores or "relevant_content" not in st.session_state.vector_stores:
            st.error("PDFs, response, and relevant content must be processed first.")
        else:
            response_text = st.session_state.vector_stores['response_text']
            context = " ".join([doc.page_content for doc in st.session_state.vector_stores["relevant_content"]])
            reference_texts = extract_reference_texts_as_json(response_text, context)
            if reference_texts is None:
                st.warning("No relevant reference texts found.")
            else:
                st.session_state.reference_texts_store["last_reference_texts"] = {"reference_texts": reference_texts}
                st.json(reference_texts)

elif st.session_state.current_page == "Generate Video Segment URLs":
    st.header("Generate Video Segment URLs")
    if st.button("Generate"):
        if "playlist_id" not in st.session_state.vector_stores or "last_reference_texts" not in st.session_state.reference_texts_store:
            st.error("Playlist and reference texts must be processed first.")
        else:
            playlist_id = st.session_state.vector_stores["playlist_id"]
            reference_texts = st.session_state.reference_texts_store["last_reference_texts"]
            filenames = [ref["filename"] for ref in reference_texts["reference_texts"]]
            videos = get_playlist_videos(playlist_id)
            relevant_video_urls = {}
            for filename in filenames:
                for video in videos:
                    if filename.lower() in video['title'].lower():
                        video_id = video['video_id']
                        relevant_video_urls[filename] = f"https://www.youtube.com/watch?v={video_id}"
            if relevant_video_urls:
                st.write(relevant_video_urls)
            else:
                st.warning("No matching video found for lessons.")

elif st.session_state.current_page == "Generate Questions":
    st.header("Generate Questions")
    question_type = st.selectbox("Select question type", ["MCQ", "True/False"])
    questions_number = st.number_input("Number of questions", min_value=1, max_value=10, step=1)
    if st.button("Generate"):
        if "last_reference_texts" not in st.session_state.reference_texts_store:
            st.error("No reference texts found. Please process the reference texts first.")
        else:
            model = genai.GenerativeModel(
                model_name="gemini-1.5-pro-latest",
                generation_config={"temperature": 0.2, "top_p": 1, "top_k": 1, "max_output_tokens": 8000},
                system_instruction="You are a helpful document answering assistant."
            )
            relevant_texts = " ".join([ref["relevant_texts"] for ref in st.session_state.reference_texts_store["last_reference_texts"]["reference_texts"]])
            questions_json = generate_questions(
                relevant_text=relevant_texts,
                num_questions=questions_number,
                question_type=question_type,
                model=model
            )
            st.json(questions_json)
