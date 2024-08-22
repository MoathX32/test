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

# Initialize session state if not already present
if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = {}
if "reference_texts_store" not in st.session_state:
    st.session_state.reference_texts_store = {}
if "document_store" not in st.session_state:
    st.session_state.document_store = []
if "navigation_visible" not in st.session_state:
    st.session_state.navigation_visible = False  # Initially hide navigation buttons
if "current_page" not in st.session_state:
    st.session_state.current_page = "Process PDFs and Videos"  # Default page

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

# Function to generate responses with both study assistance and general chat capabilities
def get_response(context, question, model):
    # Initialize chat history if not already present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Ensure chat history entries are correctly formatted
    formatted_history = []
    try:
        for entry in st.session_state.chat_history:
            if "question" in entry and "response" in entry:
                formatted_history.append({
                    "role": "user",
                    "parts": [{"text": entry["question"]}]
                })
                formatted_history.append({
                    "role": "model",  # Changed from "assistant" to "model"
                    "parts": [{"text": entry["response"]}]
                })

        # Debugging output to check the formatted history
        st.write("Formatted History:", formatted_history)

        # Always treat the question as study-related, use the provided context
        prompt_template = """
            أنت مساعد ذكي في مادة اللغة العربية للصفوف الأولى. مهمتك هي مساعدة الطلاب على فهم الدروس والإجابة على أسئلتهم باستخدام المعلومات الموجودة في الدروس .
            استخدم النص الموجود في السياق المرجعي أدناه للإجابة على السؤال. إذا لم تتمكن من العثور على إجابة في اجب من فهمك للنص، واذا كانت خارج الموضوع أخبر المستخدم أنك غير قادر على الإجابة بناءً على المعلومات المتاحة.
            لا تخصر الاجابه اكثر من اللازم
            السياق: {context}\n
            السؤال: {question}\n
            """
        chat_session = model.start_chat(history=formatted_history)

        response = chat_session.send_message(prompt_template.format(context=context, question=question))
        response_text = response.text.strip()

        # Save the current conversation in the chat history
        st.session_state.chat_history.append({"question": question, "response": response_text})

        if "لا أستطيع الإجابة" in response_text or not response_text:
            return "أنا لا أستطيع الإجابة على هذا السؤال بناءً على المعلومات المتاحة في الدروس."
        else:
            return response_text

    except KeyError as e:
        # Print detailed debug information for KeyError
        st.error(f"KeyError: Missing key {str(e)}. Here’s the problematic entry: {entry}")
        st.error(f"Formatted History: {formatted_history}")
        return f"Error: Missing key {str(e)} in the provided data. Please check the input format."
    except Exception as e:
        st.error(f"Exception occurred: {str(e)}")
        return f"حدث خطأ أثناء محاولة الإجابة على سؤالك: {str(e)}. من فضلك حاول مرة أخرى لاحقًا."




# Function to extract reference texts as JSON
def extract_reference_texts_as_json(response_text, context):
    ref_prompt = f"""
    بناءً على الإجابة التالية، حدد النص الأكثر ارتباطًا من المستندات المرجعية الخاصة بمادة اللغة العربية، والتي يجب أن تتضمن عناوين دروس مثل "درس: قواعد اللغة العربية" أو "درس: أدب العصر العباسي".
    قدم العنوان الرئيسي للدرس كمفتاح 'filename'، وأضف النص الأكثر ارتباطًا فقط تحت مفتاح 'relevant_texts'.
    الإجابة: {response_text}

    ابحث في السياق المرجعي التالي عن المعلومات التي تدعم هذه الإجابة:
    {context}

    قدم النص الأكثر ارتباطًا مع بيان مرجعه في شكل JSON كما هو موضح أعلاه.
    """

    chat_session = genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config={
            "temperature": 0.2,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 8000,
        }
    ).start_chat(history=[])

    ref_response = chat_session.send_message(ref_prompt)
    ref_response_text = ref_response.text.strip()

    # Attempt to clean and parse the text as JSON
    cleaned_text = re.sub(r'```json', '', ref_response_text).strip()
    cleaned_text = re.sub(r'```', '', cleaned_text).strip()

    try:
        reference_texts_json = json.loads(cleaned_text)
        return reference_texts_json
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing reference texts JSON: {str(e)}")
        return None

# Function to generate questions
# Function to generate questions with improved error handling
import json

import json

def generate_questions_endpoint():
    if "last_reference_texts" not in st.session_state.reference_texts_store:
        st.error("No reference texts found. Please process the reference texts first.")
        return

    try:
        # Retrieve the reference texts safely
        last_reference_texts = st.session_state.reference_texts_store.get("last_reference_texts", {})
        reference_texts = last_reference_texts.get("reference_texts", [])

        # Debugging: Output the structure of reference_texts
        st.write("Debug: Reference Texts Structure:", reference_texts)

        # Check if reference_texts is a string (indicating it might be a JSON string)
        if isinstance(reference_texts, str):
            try:
                # Attempt to parse the string as JSON
                reference_texts = json.loads(reference_texts)
            except json.JSONDecodeError:
                st.error("Failed to parse reference_texts as JSON.")
                return

        # Ensure reference_texts is now a list
        if not isinstance(reference_texts, list):
            st.error("reference_texts is not a valid list after parsing.")
            return

        # Initialize an empty list to store valid 'relevant_texts'
        relevant_texts_list = []

        # Iterate through reference_texts to extract 'relevant_texts'
        for ref in reference_texts:
            # Check if the entry is a dictionary and contains 'relevant_texts'
            if isinstance(ref, dict) and "relevant_texts" in ref:
                relevant_texts_list.append(ref["relevant_texts"])
            else:
                st.warning(f"Skipping invalid or incomplete reference text entry: {ref}")

        # Join all relevant texts into a single string
        relevant_texts = " ".join(relevant_texts_list)

        # Ensure we have valid relevant texts before proceeding
        if not relevant_texts.strip():
            st.error("No valid reference texts available for generating questions.")
            return

        # Create the model instance
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-latest",
            generation_config={"temperature": 0.2, "top_p": 1, "top_k": 1, "max_output_tokens": 8000},
            system_instruction="You are a helpful document answering assistant."
        )

        # Get user input for question generation
        questions_number = st.number_input("Number of questions", min_value=1, max_value=10, step=1)
        question_type = st.selectbox("Select question type", ["MCQ", "True/False"])

        # Generate questions based on the relevant texts
        questions_json = generate_questions(import json

import json
import streamlit as st
import google.generativeai as genai

# Other imports if needed

def generate_questions_endpoint():
    if "last_reference_texts" not in st.session_state.reference_texts_store:
        st.error("No reference texts found. Please process the reference texts first.")
        return

    try:
        # Retrieve the reference texts safely
        last_reference_texts = st.session_state.reference_texts_store.get("last_reference_texts", {})
        reference_texts = last_reference_texts.get("reference_texts", [])

        # Debugging: Output the structure of reference_texts
        st.write("Debug: Reference Texts Structure:", reference_texts)

        # Check if reference_texts is a string (indicating it might be a JSON string)
        if isinstance(reference_texts, str):
            try:
                # Attempt to parse the string as JSON
                reference_texts = json.loads(reference_texts)
            except json.JSONDecodeError:
                st.error("Failed to parse reference_texts as JSON.")
                return

        # Ensure reference_texts is now a list
        if not isinstance(reference_texts, list):
            st.error("reference_texts is not a valid list after parsing.")
            return

        # Initialize an empty list to store valid 'relevant_texts'
        relevant_texts_list = []
        filenames = []

        # Iterate through reference_texts to extract 'relevant_texts' and 'filename'
        for ref in reference_texts:
            if isinstance(ref, dict):
                # Safely extract 'relevant_texts' and 'filename' if they exist
                relevant_texts = ref.get("relevant_texts")
                filename = ref.get("filename")
                if relevant_texts:
                    relevant_texts_list.append(relevant_texts)
                if filename:
                    filenames.append(filename)
            else:
                st.warning(f"Skipping invalid or incomplete reference text entry: {ref}")

        # Join all relevant texts into a single string
        relevant_texts = " ".join(relevant_texts_list)

        # Ensure we have valid relevant texts before proceeding
        if not relevant_texts.strip():
            st.error("No valid reference texts available for generating questions.")
            return

        # Create the model instance
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-latest",
            generation_config={"temperature": 0.2, "top_p": 1, "top_k": 1, "max_output_tokens": 8000},
            system_instruction="You are a helpful document answering assistant."
        )

        # Get user input for question generation
        questions_number = st.number_input("Number of questions", min_value=1, max_value=10, step=1)
        question_type = st.selectbox("Select question type", ["MCQ", "True/False"])

        # Generate questions based on the relevant texts
        questions_json = generate_questions(
            relevant_text=relevant_texts,
            num_questions=questions_number,
            question_type=question_type,
            model=model
        )
        # Display the generated questions as JSON
        st.json(questions_json)

    except Exception as e:
        st.error(f"An error occurred while generating questions: {str(e)}")

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

    try:
        response = model.start_chat(history=[]).send_message(prompt_template)
        response_text = response.text.strip()

        if response_text:
            response_json = json.loads(response_text)
            return response_json
        else:
            st.warning("Received an empty response from the model.")
            return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Function to get playlist videos (mock implementation)
def get_playlist_videos(playlist_id):
    # Mock implementation
    # This should be replaced with actual code that interacts with YouTube API to fetch playlist videos
    return [
        {"title": "حقوقي وواجباتي في البيت", "video_id": "abc123"},
        {"title": "كيف تنتخب مجلس القسم؟", "video_id": "def456"},
        {"title": "كيف نمارس مواطنتنا في المدرسة؟", "video_id": "ghi789"}
    ]

# Main App Interface
st.title("مرحبا ! انا مساعدك الذكي في ماددة اللغه العربيه للصف الرابع الابتدائي")

# Button to show/hide navigation
if st.button("Toggle Navigation"):
    st.session_state.navigation_visible = not st.session_state.navigation_visible

# Conditional Navigation Buttons
if st.session_state.navigation_visible:
    if st.button("Process PDFs and Videos"):
        st.session_state.current_page = "Process PDFs and Videos"
    if st.button("Generate Response"):
        st.session_state.current_page = "Generate Response"
    if st.button("Generate Reference Texts"):
        st.session_state.current_page = "Generate Reference Texts"
    if st.button("Generate Video Segment URLs"):
        st.session_state.current_page = "Generate Video Segment URLs"
    if st.button("Generate Questions"):
        st.session_state.current_page = "Generate Questions"

# Page Routing
if st.session_state.current_page == "Process PDFs and Videos":
    st.header("ابدأ")
    if st.button("Process"):
        pdf_docs_with_names = read_files_from_folder(FOLDER_PATH)
        documents = get_all_pdfs_chunks(pdf_docs_with_names)
        pdf_vectorstore = get_vector_store(documents)
        playlist_id = PLAYLIST_URL.split("list=")[-1].split("&")[0]
        st.session_state.vector_stores["pdf_vectorstore"] = pdf_vectorstore
        st.session_state.vector_stores["playlist_id"] = playlist_id
        st.session_state.document_store.extend(documents)
        st.success("PDFs and playlist processed successfully!")
        st.session_state.navigation_visible = True  # Show navigation after successful processing

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
