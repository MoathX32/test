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
    st.session_state.chat_history = []  # Initialize chat history

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
    # Prepare the prompt by including the previous conversation history
    prompt_template = """
    أنت مساعد ذكي في مادة اللغة العربية للصفوف الأولى. تفهم أساسيات اللغة العربية مثل الحروف، الكلمات البسيطة، والجمل الأساسية.
    أجب على السؤال التالي باستخدام النص الموجود في السياق المرجعي أدناه فقط. قدم إجابة بسيطة وواضحة تتناسب مع مستوى الصفوف الأولى.
    يجب أن يكون الرد مختصرًا ومفهومًا.
    لا تجب على أي سؤال خارج سياق النص.
    
    السياق: {context}\n
    السؤال: {question}\n
    """

    # Initialize or update the chat history
    chat_history = st.session_state.get("chat_history", [])
    chat_history.append({"role": "user", "content": question})

    chat_session = model.start_chat(history=chat_history)

    try:
        # Send the message and get the response
        response = chat_session.send_message(prompt_template.format(context=context, question=question))
        response_text = response.text

        # Update chat history with the bot's response
        chat_history.append({"role": "assistant", "content": response_text})
        st.session_state.chat_history = chat_history

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
        model_name="gemini-1.5-pro-latest",
        generation_config=generation_config,
        system_instruction="You are a helpful document answering assistant."
    )
    
    response = get_response(context, query_request.query, model)
    st.session_state.vector_stores["response_text"] = response  # Store the response for later use
    return response

def extract_reference_texts_as_json(response_text, context):
    ref_prompt = f"""
    بناءً على الإجابة التالية، حدد النص الأكثر ارتباطًا من المستندات المرجعية الخاصة بمادة اللغة العربية، والتي يجب أن تتضمن عناوين دروس مثل "درس: قواعد اللغة العربية" أو "درس: أدب العصر العباسي".
    قدم العنوان الرئيسي للدرس كمفتاح 'filename'، وأضف النص الأكثر ارتباطًا فقط تحت مفتاح 'relevant_texts'.
    
    يجب أن يكون الإخراج بتنسيق JSON يتضمن 'filename' و 'relevant_texts' كما هو موضح:
    [
        {{
            "filename": "عنوان الدرس",
            "relevant_texts": "النص الأكثر ارتباطًا فقط"
        }}
    ]

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

    try:
        reference_texts_json = json.loads(ref_response_text)
    except json.JSONDecodeError:
        reference_texts_json = None
    
    return reference_texts_json

def generate_reference_texts():
    if "pdf_vectorstore" not in st.session_state.vector_stores or "response_text" not in st.session_state.vector_stores or "relevant_content" not in st.session_state.vector_stores:
        raise HTTPException(status_code=400, detail="PDFs, response, and relevant content must be processed first.")
    
    response_text = st.session_state.vector_stores['response_text']

    context = " ".join([doc.page_content for doc in st.session_state.vector_stores["relevant_content"]])

    reference_texts = extract_reference_texts_as_json(response_text, context)
    
    if reference_texts is None:
        logging.warning("No relevant reference texts found.")
        st.session_state.reference_texts_store["last_reference_texts"] = None
    else:
        st.session_state.reference_texts_store["last_reference_texts"] = {"reference_texts": reference_texts}
    
    return reference_texts

def find_video_segment(filenames, response_text, playlist_id):
    videos = get_playlist_videos(playlist_id)
    relevant_video_urls = {}

    for filename in filenames:
        for video in videos:
            if filename.lower() in video['title'].lower():
                video_id = video['video_id']
                relevant_video_urls[filename] = f"https://www.youtube.com/watch?v={video_id}"
                break

    if not relevant_video_urls:
        logging.warning(f"No matching video found for lessons: {filenames}")
        return None

    return relevant_video_urls

def generate_video_segment_url():
    if "playlist_id" not in st.session_state.vector_stores or "last_reference_texts" not in st.session_state.reference_texts_store:
        logging.error("Required data not found: playlist_id or last_reference_texts.")
        raise HTTPException(status_code=400, detail="Playlist and reference texts must be processed first.")
    
    if st.session_state.reference_texts_store.get("last_reference_texts") is None:
        logging.error("Cannot generate video segment URLs because reference texts are None.")
        raise HTTPException(status_code=400, detail="Reference texts are None. Cannot generate video segment URLs.")

    playlist_id = st.session_state.vector_stores.get("playlist_id")
    reference_texts = st.session_state.reference_texts_store.get("last_reference_texts")
    
    filenames = [ref["filename"] for ref in reference_texts["reference_texts"]]
    response_text = st.session_state.vector_stores["response_text"]

    if not filenames:
        logging.error("Lesson names are missing from the reference texts.")
        return None

    video_segment_urls = find_video_segment(filenames, response_text, playlist_id)
    
    if not video_segment_urls:
        logging.error(f"Video segments not found for lessons: {filenames}.")
        raise HTTPException(status_code=404, detail="Not Found")

    return video_segment_urls

class QuestionRequest(BaseModel):
    question_type: str
    questions_number: int

def generate_questions(relevant_text, num_questions, question_type, model):
    if not relevant_text.strip():
        logging.warning("Relevant text is empty or invalid.")
        st.error("Relevant text is empty or invalid.")
        return None

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

        logging.info(f"Model Response: {response_text}")  # Log the model's response

        try:
            response_json = json.loads(response_text)
        except json.JSONDecodeError:
            response_json = None

        return response_json
    except Exception as e:
        logging.warning(f"Error: {e}")
        st.error(f"Error generating questions: {e}")
        return None

def generate_questions_endpoint(question_request: QuestionRequest):
    if "last_reference_texts" not in st.session_state.reference_texts_store:
        raise HTTPException(status_code=400, detail="No reference texts found. Please process the reference texts first.")
    
    if st.session_state.reference_texts_store.get("last_reference_texts") is None:
        logging.error("Cannot generate questions because reference texts are None.")
        raise HTTPException(status_code=400, detail="Reference texts are None. Cannot generate questions.")

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

    relevant_texts = " ".join([ref["relevant_texts"] for ref in st.session_state.reference_texts_store["last_reference_texts"]["reference_texts"]])

    questions_json = generate_questions(
        relevant_text=relevant_texts,
        num_questions=question_request.questions_number,
        question_type=question_request.question_type,
        model=model
    )

    return questions_json

def get_playlist_videos(playlist_id):
    # Mock implementation
    # This should be replaced with actual code that interacts with YouTube API to fetch playlist videos
    return [
        {"title": "حقوقي وواجباتي في البيت", "video_id": "abc123"},
        {"title": "كيف تنتخب مجلس القسم؟", "video_id": "def456"},
        {"title": "كيف نمارس مواطنتنا في المدرسة؟", "video_id": "ghi789"}
    ]

# Streamlit UI Components
import streamlit as st

# التحقق من وجود الحالات في session_state
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

if "response_submitted" not in st.session_state:
    st.session_state.response_submitted = False

if "sources_shown" not in st.session_state:
    st.session_state.sources_shown = False

# عنوان الصفحة
st.title("مرحبا بك! أنا مساعد مادة اللغة العربية للصف الرابع")

# إرشادات الاستخدام
with st.expander("إرشادات الاستخدام"):
    st.write("""
    **تنبيه:**
    البرنامج مازال تحت التجريب وقد يحتوي على بعض الأخطاء أو الميزات غير المكتملة. نقدر تفهمك وأي ملاحظات قد تساعد في تحسين الأداء.

    **إرشادات المستخدم لواجهة Streamlit:**
    1. **تشغيل المساعد:**
       عند فتح واجهة Streamlit، ستجد زرًا بعنوان "ابدأ تشغيل المساعد". بالضغط على هذا الزر، يبدأ البرنامج في معالجة ملفات PDF الموجودة في مجلد Data وتحليل محتوى الفيديوهات من قائمة تشغيل YouTube المحددة.
    2. **طرح سؤال:**
       في الجزء المخصص للأسئلة، يمكنك إدخال سؤالك في حقل النص "كيف يمكنني مساعدتك". بعد إدخال السؤال، اضغط على زر "أجب". سيقوم البرنامج بمعالجة سؤالك بناءً على النصوص المستخرجة من ملفات PDF ويعرض الرد في الأسفل.
    3. **عرض المصادر:**
       بعد الحصول على الرد على سؤالك، يمكنك الضغط على زر "المصادر" لعرض النصوص المرجعية من ملفات PDF التي تم استخدامها لتكوين الإجابة.
    4. **إنشاء أسئلة اختبار:**
       في قسم إنشاء الأسئلة، اختر نوع السؤال الذي ترغب في إنشائه (اختيارات متعددة "MCQ" أو صح/خطأ "True/False").
       حدد عدد الأسئلة باستخدام المؤشر، ثم اضغط على "ابدأ وضع الاختبار". سيتم عرض الأسئلة المتولدة بناءً على النصوص المرجعية.
    5. **إنشاء روابط مقاطع الفيديو:**
       إذا كنت قد قمت بمعالجة النصوص المرجعية وتحتاج إلى العثور على المقاطع المتعلقة بهذه النصوص في قائمة تشغيل YouTube، يمكنك الضغط على زر "Generate Video Segment URLs". سيقوم البرنامج بتوليد الروابط للمقاطع ذات الصلة ويعرضها لك.
    6. **تنبيهات وأخطاء:**
       إذا واجهتك أي أخطاء، ستظهر رسائل خطأ في الواجهة تشرح المشكلة، مثل عدم العثور على ملفات PDF، أو عدم القدرة على توليد الردود أو الأسئلة.
    """)

st.write("---")

# إضافة مساحات فارغة أعلى الصفحة لتوسيط الزر عموديًا
st.write("")
st.write("")
st.write("")

# زر "ابدأ تشغيل المساعد"
if st.button('🚀 ابدأ تشغيل المساعد 🚀'):
    # إعادة تعيين المتغيرات في session_state لمسح سجل الدردشة
    st.session_state.vector_stores = {}
    st.session_state.reference_texts_store = {}
    st.session_state.document_store = []
    st.session_state.response_submitted = False
    st.session_state.sources_shown = False
    st.session_state.chat_history = []  # مسح سجل الدردشة

    with st.spinner('جاري معالجة الملفات...'):
       process_lessons_and_video()  # استدعاء الدالة لمعالجة الملفات
    st.session_state.processing_complete = True  # تحديث حالة المعالجة

st.write("---")

# إظهار النموذج response_form فقط إذا تمت معالجة الملفات
if st.session_state.processing_complete:
    with st.form(key='response_form'):
        query = st.text_input("كيف يمكنني مساعدتك:")
        response_button = st.form_submit_button(label='أجب')

        if response_button:
            query_request = QueryRequest(query=query)
            response = generate_response(query_request)
            st.write("الرد:", response)
            st.session_state.response_submitted = True  # تحديث حالة تقديم الرد

    st.write("---")

    # إظهار زر المصادر فقط بعد تقديم الرد
    if st.session_state.response_submitted:
        if st.session_state.get("vector_stores") and st.button("المصادر"):
            reference_texts = generate_reference_texts()
            st.write("النصوص من الكتاب:", reference_texts)
            st.session_state.sources_shown = True  # تحديث حالة عرض المصادر

        st.write("---")

        # إظهار باقي العناصر بعد عرض المصادر
        if st.session_state.sources_shown:
            # نموذج لإنشاء الأسئلة
            with st.form(key='questions_form'):
                question_type = st.selectbox("اختر نوع السؤال:", ["MCQ", "True/False"])
                questions_number = st.number_input("اختر عدد الأسئلة:", min_value=1, max_value=10)
                generate_questions_button = st.form_submit_button(label='ابدأ وضع الاختبار')

                if generate_questions_button:
                    question_request = QuestionRequest(question_type=question_type, questions_number=questions_number)
                    questions = generate_questions_endpoint(question_request)
                    st.write("الاختبار:", questions)

            st.write("---")

            # زر لتوليد روابط مقاطع الفيديو
            if st.session_state.get("reference_texts_store") and st.button("Generate Video Segment URLs"):
                video_segment_urls = generate_video_segment_url()
                st.write("Generated Video Segment URLs:", video_segment_urls)
