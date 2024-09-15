
# Question Generation and Management System

This project is an advanced question generation and management system built using FastAPI, Google Generative AI, SQLite, and other powerful tools. The system is designed to extract content from PDF files, generate relevant questions, and manage them efficiently. It provides an interactive interface for generating questions, reviewing modifications, and managing the lifecycle of questions, making it an ideal solution for educators and institutions.

## Key Features

1. **Automated Question Generation**:
   - Utilizes Google Generative AI to generate multiple-choice questions (MCQs), true/false questions, and open-ended written questions from the content extracted from PDF documents.
   - Supports varying levels of difficulty and incorporates real-life scenarios to enhance comprehension.

2. **PDF Content Extraction**:
   - Extracts and processes text from PDF files using PyPDF2.
   - Splits the content into manageable chunks for generating questions.

3. **Database Management**:
   - Stores questions in a SQLite database with detailed metadata, including lesson names, question types, options, and correct answers.
   - Manages a separate reviewed questions database to track modifications and approvals.

4. **Question Modification and Approval**:
   - Allows educators to modify questions, including the text, options, and correct answers, with an option to provide reasons for the changes.
   - Provides a mechanism for supervisors to review and approve or reject modifications, ensuring quality and accuracy.

5. **Automated Feedback System**:
   - Sends feedback to the AI model based on reviewed questions, allowing the model to learn from corrections and improve future outputs.

6. **FastAPI Integration**:
   - Provides a RESTful API interface for generating questions, submitting modifications, and approving changes.
   - Supports batch processing of PDFs and questions, making it scalable and efficient.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the project root and add your Google Generative AI API key:
     ```
     GOOGLE_API_KEY=<your_api_key>
     ```

4. Run the application:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## Usage

1. **Generate Questions**: 
   - Upload PDF files to a specified directory and call the `/generate-questions/` endpoint to generate questions.

2. **Modify Questions**:
   - Use the `/modify-question/` endpoint to submit modifications, deletions, or updates to existing questions.

3. **Supervisor Approval**:
   - Supervisors can approve or reject modifications using the `/supervise-questions/` endpoint.

4. **Apply Feedback**:
   - Use the `/apply-feedback/` endpoint to update the model with the reviewed and approved questions, ensuring continuous learning.

## API Endpoints

### 1. Generate Questions

**Endpoint:** `/generate-questions/`

**Method:** `POST`

**Description:** Generates questions from PDF files located in a specified directory.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/generate-questions/" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "pdf_directory=./pdfs" \
    -d "question_counts={\"Lesson1\": 5, \"Lesson2\": 3}" \
    -d "question_types={\"Lesson1\": \"MCQ\", \"Lesson2\": \"TF\"}"
```

**Parameters:**
- `pdf_directory`: Path to the directory containing the PDF files.
- `question_counts`: A JSON string specifying the number of questions to generate per lesson.
- `question_types` (optional): A JSON string specifying the question type for each lesson (e.g., MCQ, TF, WRITTEN).

**Response:**
```json
{
    "Lesson1.pdf": [
        {
            "question": "What is the primary function of a database?",
            "options": ["Store data", "Delete data", "Format data", "Transfer data"],
            "correct_answer": "Store data"
        }
    ],
    "Lesson2.pdf": [
        {
            "question": "True or False: SQL is used to manipulate data in a database.",
            "options": ["صواب", "خطأ"],
            "correct_answer": "صواب"
        }
    ]
}
```


![](https://github.com/expotech-online/EduAI-DL/blob/main/Final_Models/Final_QG_Model/Screenshot%20(321).png)

### 2. Modify Questions

**Endpoint:** `/modify-question/`

**Method:** `POST`

**Description:** Modifies an existing question, with the option to delete the question or provide a reason for modification.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/modify-question/" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "question_id=1" \
    -d "modified_question=What is a database?" \
    -d "modified_options=[\"A software\", \"A hardware\", \"An accessory\", \"None\"]" \
    -d "modified_answer=A software" \
    -d "modification_reason=Rephrased for clarity" \
    -d "delete=false"
```

**Parameters:**
- `question_id`: ID of the question to modify.
- `modified_question` (optional): The new question text.
- `modified_options` (optional): The new options in JSON string format.
- `modified_answer` (optional): The new correct answer.
- `modification_reason`: Reason for the modification or deletion.
- `delete`: Boolean flag to delete the question.

**Response:**
```json
{
    "status": "Modification request logged and pending supervisor approval."
}
```

![](https://github.com/expotech-online/EduAI-DL/blob/main/Final_Models/Final_QG_Model/Screenshot%20(330).png)

### 3. Supervisor Approval

**Endpoint:** `/supervise-questions/`

**Method:** `POST`

**Description:** Allows a supervisor to approve or reject modifications or deletions to questions.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/supervise-questions/" \
    -H "Content-Type: application/json" \
    -d '{
        "decisions": [
            {"question_id": 1, "approval_status": true},
            {"question_id": 2, "approval_status": false}
        ]
    }'
```

**Parameters:**
- `decisions`: A list of decisions, each containing:
  - `question_id`: ID of the question.
  - `approval_status`: `true` to approve, `false` to reject.

**Response:**
```json
{
    "status": "Batch processing of questions completed successfully."
}
```

![](https://github.com/expotech-online/EduAI-DL/blob/main/Final_Models/Final_QG_Model/Screenshot%20(331).png)

### 4. Apply Feedback

**Endpoint:** `/apply-feedback/`

**Method:** `POST`

**Description:** Updates the AI model with reviewed and approved questions, reinforcing learning.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/apply-feedback/" \
    -H "Content-Type: application/json"
```

**Response:**
```json
{
    "status": "Model updated with reviewed questions successfully."
}
```


