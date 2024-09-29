import os
import io
import json
import logging
import streamlit as st
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List
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
    أنت مساعد ذكي في مادة اللغة العربية للصفوف الأولى. تفهم أساسيات اللغة العربية مثل الحروف، الكلمات البسيطة، والجمل الأساسية.
    لاتجب الا اذا كان السؤال واضح او استفهم من الطالب المقصود
    أجب على السؤال التالي من خلال فهمك النص الموجود في السياق المرجعي . قدم إجابة بسيطة وواضحة تتناسب مع مستوى الصفوف الأولى.
    يجب أن يكون الرد مفهومًا.
    لا تجب على أي سؤال خارج فهمك سياق النص.
    السياق: {context}\n
    السؤال: {question}\n
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
        model_name="gemini-1.5-pro-latest",
        generation_config=generation_config,
        system_instruction="You are a helpful document answering assistant."
    )
    
    response = get_response(context, query_request.query, model)
    st.session_state.vector_stores["response_text"] = response  # Store the response for later use
    return response

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

def extract_reference_texts_as_json(response_text, context):
    ref_prompt = f"""
    بناءً على الإجابة التالية، حدد النص الأكثر ارتباطًا من المستندات المرجعية الخاصة بمادة اللغة العربية.
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

    # تسجيل النص المستلم لفحصه
    logging.info(f"Reference response text: {ref_response_text}")

    reference_texts_json = clean_json_response(ref_response_text)
    
    if reference_texts_json is None:
        logging.warning("Failed to parse JSON from reference response.")
    else:
        logging.info(f"Parsed reference texts JSON: {reference_texts_json}")
    
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
        You are an AI assistant tasked with generating exactly {num_questions} multiple-choice questions (MCQs) from the given context. 
        Create a set of MCQs with 4 answer options each. Ensure that the questions cover key concepts from the context provided.
        The questions should be formatted in JSON with fields 'question', 'options', and 'correct_answer'.
        
        Context: {relevant_text}\n
        """
    else:
        prompt_template = f"""
        You are an AI assistant tasked with generating exactly {num_questions} true/false questions from the given context. 
        The questions should be formatted in JSON with fields 'question' and 'correct_answer'. 
        
        Context: {relevant_text}\n
        """

    try:
        response = model.start_chat(history=[]).send_message(prompt_template)
        response_text = response.text.strip()

        logging.info(f"Model Response: {response_text}")

        if response_text:
            response_json = clean_json_response(response_text)
            return response_json if response_json else None
        else:
            logging.warning("Received an empty response from the model.")
            st.error("Received an empty response from the model.")
            return None
    except Exception as e:
        logging.warning(f"Error: {e}")
        st.error(f"Error generating questions: {e}")
        return None
    
def generate_questions_endpoint(question_request: QuestionRequest):
    if "last_reference_texts" not in st.session_state.reference_texts_store:
        raise HTTPException(status_code=400, detail="No reference texts found. Please process the reference texts first.")
    
    reference_texts = st.session_state.reference_texts_store.get("last_reference_texts", {})
    
    # Ensure that the reference texts are properly extracted
    if "reference_texts" in reference_texts and isinstance(reference_texts["reference_texts"], dict):
        relevant_texts = reference_texts["reference_texts"].get("relevant_texts", "")
        
        if not relevant_texts.strip():
            logging.error("Relevant texts are empty or invalid.")
            st.error("النصوص المرجعية غير صحيحة.")
            return None

        logging.info(f"Relevant texts extracted: {relevant_texts}")
    else:
        st.error("النصوص المرجعية غير صحيحة.")
        logging.error(f"Reference texts structure: {reference_texts}")
        return None

    questions_json = generate_questions(
        relevant_text=relevant_texts,
        num_questions=question_request.questions_number,
        question_type=question_request.question_type,
        model=genai.GenerativeModel(
            model_name="gemini-1.5-pro-latest",
            generation_config={
                "temperature": 0.2,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 8000,
            },
            system_instruction="You are a helpful document answering assistant."
        )
    )

    return questions_json

# Streamlit UI Components
import streamlit as st

# عنوان الصفحة
st.title("مرحبا بك! أنا مساعد مادة اللغة العربية للصف الرابع")

# إرشادات الاستخدام
with st.expander("إرشادات الاستخدام"):
    st.write("""
    **تنبيه:**
    البرنامج مازال تحت التجريب وقد يحتوي على بعض الأخطاء أو الميزات غير المكتملة. نقدر تفهمك وأي ملاحظات قد تساعد في تحسين الأداء.

    **إرشادات المستخدم لواجهة Streamlit:**
    1. **تشغيل المساعد:**
       عند فتح واجهة Streamlit، ستجد زرًا بعنوان "ابدأ تشغيل المساعد". بالضغط على هذا الزر، يبدأ البرنامج في معالجة ملفات PDF الموجودة في مجلد Data.
    2. **طرح سؤال:**
       في الجزء المخصص للأسئلة، يمكنك إدخال سؤالك في حقل النص "كيف يمكنني مساعدتك". بعد إدخال السؤال، اضغط على زر "أجب". سيقوم البرنامج بمعالجة سؤالك بناءً على النصوص المستخرجة من ملفات PDF ويعرض الرد في الأسفل.
    3. **إنشاء أسئلة اختبار:**
       في قسم إنشاء الأسئلة، اختر نوع السؤال الذي ترغب في إنشائه (اختيارات متعددة "MCQ" أو صح/خطأ "True/False").
       حدد عدد الأسئلة باستخدام المؤشر، ثم اضغط على "ابدأ وضع الاختبار". سيتم عرض الأسئلة المتولدة بناءً على النصوص المرجعية.
    """)

st.write("---")

# إضافة مساحات فارغة أعلى الصفحة لتوسيط الزر عموديًا
st.write("")
st.write("")
st.write("")


# استخدام st.button مع نفس النص لتقديم نفس الوظيفة
if st.button('🚀 ابدأ تشغيل المساعد 🚀'):
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

    # بعد تقديم الرد، استخراج النصوص المرجعية في الخلفية
    if st.session_state.response_submitted:
        reference_texts = generate_reference_texts()
        if reference_texts is not None:
            st.session_state.sources_shown = True  # تحديث حالة استخراج المصادر

    st.write("---")

    # إظهار باقي العناصر بعد تقديم الرد
    if st.session_state.sources_shown:
        # نموذج لإنشاء الأسئلة
        with st.form(key='questions_form'):
            question_type = st.selectbox("اختر نوع السؤال:", ["MCQ", "True/False"])
            questions_number = st.number_input("اختر عدد الأسئلة:", min_value=1, max_value=30)
            generate_questions_button = st.form_submit_button(label='ابدأ وضع الاختبار')

            if generate_questions_button:
                question_request = QuestionRequest(question_type=question_type, questions_number=questions_number)
                questions = generate_questions_endpoint(question_request)
                st.write("الاختبار:", questions)
