import os
from fastapi import FastAPI, Form, HTTPException
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
import logging
import io
import json

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
genai_api_key = os.getenv("GENAI_API_KEY")

# Configure GenAI
genai.configure(api_key=genai_api_key)

app = FastAPI()

vector_stores: Dict[str, any] = {}
reference_texts_store: Dict[str, str] = {}
document_store: List[Document] = []  # To store original documents

# Step 1: Process PDFs and Videos
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

@app.post("/process")
async def process_lessons_and_video(folder_path: str = Form(...), playlist_url: str = Form(...)):
    pdf_docs_with_names = read_files_from_folder(folder_path)
    if not pdf_docs_with_names or any(len(pdf) == 0 for pdf, _ in pdf_docs_with_names):
        raise HTTPException(status_code=400, detail="One or more PDF files are empty.")

    documents = get_all_pdfs_chunks(pdf_docs_with_names)
    pdf_vectorstore = get_vector_store(documents)

    playlist_id = playlist_url.split("list=")[-1].split("&")[0]

    vector_stores["pdf_vectorstore"] = pdf_vectorstore
    vector_stores["playlist_id"] = playlist_id
    document_store.extend(documents)  # Store original documents

    return {"message": "PDFs and playlist processed successfully", "playlist_id": playlist_id}

# Step 2: Generate Response
class QueryRequest(BaseModel):
    query: str

def get_response(context, question, model):
    chat_session = model.start_chat(history=[])

    prompt_template = """
    أنت مساعد ذكي في مادة اللغة العربية للصفوف الأولى. تفهم أساسيات اللغة العربية مثل الحروف، الكلمات البسيطة، والجمل الأساسية.
    أجب على السؤال التالي باستخدام النص الموجود في السياق المرجعي أدناه فقط. قدم إجابة بسيطة وواضحة تتناسب مع مستوى الصفوف الأولى.
    يجب أن يكون الرد مختصرًا ومفهومًا.
    لا تجب على أي سؤال خارج سياق النص.
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

@app.post("/response")
@app.post("/response")
async def generate_response(query_request: QueryRequest):
    if "pdf_vectorstore" not in vector_stores:
        raise HTTPException(status_code=400, detail="PDFs must be processed first using the /process endpoint.")
    
    pdf_vectorstore = vector_stores['pdf_vectorstore']
    
    # البحث عن المحتوى ذو الصلة بالسؤال
    relevant_content = pdf_vectorstore.similarity_search(query_request.query, k=20)
    
    # حفظ المحتوى ذو الصلة لاستخدامه لاحقًا في استخراج النصوص المرجعية
    vector_stores["relevant_content"] = relevant_content

    # تحويل المحتوى إلى نصوص لاستخدامها في توليد الرد
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
    vector_stores["response_text"] = response  # Store the response for later use
    return {"response": response}

# Step 3: Extract Reference Texts
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

    # إرسال القالب إلى النموذج للحصول على النصوص المرجعية في شكل JSON
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

    # محاولة تنظيف وتحليل النص كـ JSON
    reference_texts_json = clean_json_response(ref_response_text)
    
    return reference_texts_json
@app.post("/reference_texts")
async def generate_reference_texts():
    if "pdf_vectorstore" not in vector_stores or "response_text" not in vector_stores or "relevant_content" not in vector_stores:
        raise HTTPException(status_code=400, detail="PDFs, response, and relevant content must be processed first using the /process and /response endpoints.")
    
    response_text = vector_stores['response_text']
    
    # التحقق مما إذا كان response_text يحتوي على إجابة غير صالحة أو فارغة
    invalid_phrases = [
        "لا يمكنني الإجابة", 
        "النص غير مكتمل", 
        "السؤال غير واضح", 
        "خارج المنهج", 
        "غير مرتبط", 
        "تصحيح", 
        "مرحبا", 
        "أهلا", 
        "السلام عليكم", 
        "تحية", 
        "صباح الخير", 
        "مساء الخير", 
        "غير قادر", 
        "لم أتمكن", 
        "لا أستطيع", 
        "غير مفهوم", 
        "لم أتمكن من الفهم", 
        "لا أفهم", 
        "سؤال ناقص", 
        "سؤال غير مكتمل", 
        "لا تتعلق", 
        "هذا ليس جزءًا من", 
        "لم يتم العثور على", 
        "لا يوجد معلومات",
        "لم تحدد",
        "هل يمكنك",
        "لا استطيع" ,
        "غير قادر",
        "لم يذكر",
        "لم تتحدث",
        "من فضلك",
        "لا يوجد",
        "السياق المذكور",
        "لم تحدد",
        "لا يتضمن إجابة"
    ]

    if any(phrase in response_text for phrase in invalid_phrases):
        logging.warning("The response_text is invalid or incomplete.")
        reference_texts_store["last_reference_texts"] = None
        return {"reference_texts": None}

    # استخدام المحتوى المخزن مسبقًا لاستخراج النصوص المرجعية
    context = " ".join([doc.page_content for doc in vector_stores["relevant_content"]])

    reference_texts = extract_reference_texts_as_json(response_text, context)
    
    if reference_texts is None:
        logging.warning("No relevant reference texts found.")
        reference_texts_store["last_reference_texts"] = None
    else:
        reference_texts_store["last_reference_texts"] = {"reference_texts": reference_texts}
    
    return {"reference_texts": reference_texts}

# Step 4: Generate Video Segment URLs
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

@app.post("/video_segment_url")
async def generate_video_segment_url():
    if "playlist_id" not in vector_stores or "last_reference_texts" not in reference_texts_store:
        logging.error("Required data not found: playlist_id or last_reference_texts.")
        raise HTTPException(status_code=400, detail="Playlist and reference texts must be processed first using the /process and /reference_texts endpoints.")
    
    # التحقق مما إذا كانت reference_texts تساوي None
    if reference_texts_store.get("last_reference_texts") is None:
        logging.error("Cannot generate video segment URLs because reference texts are None.")
        raise HTTPException(status_code=400, detail="Reference texts are None. Cannot generate video segment URLs.")

    playlist_id = vector_stores.get("playlist_id")
    reference_texts = reference_texts_store.get("last_reference_texts")
    
    filenames = [ref["filename"] for ref in reference_texts["reference_texts"]]
    response_text = vector_stores["response_text"]

    if not filenames:
        logging.error("Lesson names are missing from the reference texts.")
        return {"video_segment_url": None}

    video_segment_urls = find_video_segment(filenames, response_text, playlist_id)
    
    if not video_segment_urls:
        logging.error(f"Video segments not found for lessons: {filenames}.")
        raise HTTPException(status_code=404, detail="Not Found")

    return {"video_segment_urls": video_segment_urls}

# Step 5: Generate Questions
class QuestionRequest(BaseModel):
    question_type: str
    questions_number: int

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

        logging.info(f"Model Response: {response_text}")  # تسجيل استجابة النموذج

        if response_text:
            response_json = clean_json_response(response_text)
            if response_json:
                return response_json
            else:
                logging.warning("Failed to decode JSON from model response.")
                return None
        else:
            logging.warning("Received an empty response from the model.")
            return None
    except Exception as e:
        logging.warning(f"Error: {e}")
        return None

@app.post("/generate_questions")
async def generate_questions_endpoint(question_request: QuestionRequest):
    if "last_reference_texts" not in reference_texts_store:
        raise HTTPException(status_code=400, detail="No reference texts found. Please process the reference texts first using the /reference_texts endpoint.")
    
    # التحقق مما إذا كانت reference_texts تساوي None
    if reference_texts_store.get("last_reference_texts") is None:
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

    relevant_texts = " ".join([ref["relevant_texts"] for ref in reference_texts_store["last_reference_texts"]["reference_texts"]])

    questions_json = generate_questions(
        relevant_text=relevant_texts,
        num_questions=question_request.questions_number,
        question_type=question_request.question_type,
        model=model
    )

    return {"generated_questions": questions_json}

# Helper Function to Get Playlist Videos
def get_playlist_videos(playlist_id):
    # Mock implementation
    # This should be replaced with actual code that interacts with YouTube API to fetch playlist videos
    return [
        {"title": "حقوقي وواجباتي في البيت", "video_id": "abc123"},
        {"title": "كيف تنتخب مجلس القسم؟", "video_id": "def456"},
        {"title": "كيف نمارس مواطنتنا في المدرسة؟", "video_id": "ghi789"}
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
