# 📚 Domain-Specific RAG Chatbot for PDF Question Answering

A production-ready **Retrieval-Augmented Generation (RAG)** Streamlit application that enables users to upload PDF documents (such as employee handbooks, manuals, policies, or course notes) and receive grounded, accurate answers backed by **exact source document and page number citations**.

---

## 1. Problem Statement
Large PDF documents are difficult to navigate manually. Searching for specific details across multi-page manuals is slow and error-prone. Standard keyword searches fail on natural language queries, while general LLMs hallucinate when asked about private domain documents. This project solves this by retrieving relevant text passages and generating answers grounded strictly in the uploaded PDF content.

---

## 2. Project Objective
* **Extract & Tag Metadata**: Read PDF pages using `pypdf` and attach document name and 1-indexed page number metadata.
* **Chunk & Embed**: Split documents using LangChain's `RecursiveCharacterTextSplitter` and generate dense vector embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
* **FAISS Vector Database**: Index embeddings using FAISS and persist index files to disk (`vector_store/saved_index/`).
* **Grounded LLM Answers**: Query Groq LLM using strict system prompts to eliminate hallucinations and defend against document prompt injection.
* **Source Attribution**: Display exact document names and page numbers for every generated answer.
* **Fallback Refusal**: Return the exact string `"I could not find this information in the uploaded documents."` when information is absent.

---

## 3. How the RAG Workflow Works

```text
Upload PDFs ──► Extract Text (pypdf) ──► Page Metadata ──► Chunking (LangChain)
                                                                │
Answer + Source/Page ◄── Groq LLM ◄── Prompt Guardrail ◄── FAISS Top-k Search ◄── Embedding
```

1. **Document Upload**: User uploads one or multiple PDF files (Max 25MB per file) via the Streamlit sidebar.
2. **Text Extraction**: `document_loader.py` reads pages using `pypdf`, capturing `source` filename and `page_number` while skipping blank pages.
3. **Chunking**: `vector_store.py` splits text using `RecursiveCharacterTextSplitter` (chunk size ~800, overlap ~120).
4. **Vector Embedding**: Chunks are embedded into 384-dimensional vectors using `all-MiniLM-L6-v2`.
5. **FAISS Storage**: Vectors and metadata are stored in a FAISS index and saved to `vector_store/saved_index/`.
6. **Query & Retrieval**: User questions are converted to embeddings to search FAISS for top 3–5 relevant chunks.
7. **Prompt Guardrail & Generation**: The prompt combines context and safeguards against injection. Groq LLM generates the response.
8. **Attribution / Refusal**: Answer is displayed alongside expandable source/page accordions, or exact fallback refusal.

---

## 4. Technology Stack
* **Language**: Python 3.14
* **PDF Processing**: `pypdf`
* **Text Chunking**: LangChain (`langchain-text-splitters`)
* **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
* **Vector Database**: `faiss-cpu`
* **LLM Engine**: Groq API (`llama-3.3-70b-versatile` / `llama3-8b-8192`)
* **Frontend**: Streamlit
* **Environment**: `python-dotenv`

---

## 5. Repository Structure
```text
finla_AI(repo)/
│
├── app.py                     # Streamlit frontend with sidebar & chat controls
├── rag_pipeline.py            # End-to-end RAG workflow orchestration
├── document_loader.py         # PDF text extraction, validation & page metadata
├── vector_store.py            # LangChain chunking, embeddings, FAISS & persistence
├── prompt.py                  # System guardrails, exact fallback & injection defense
├── requirements.txt           # Python package dependencies
├── README.md                  # Project setup and usage documentation
├── PROJECT_REPORT.md          # Comprehensive report & Viva QA guide
├── .env                       # Local API key configuration (Git-ignored)
├── .gitignore                 # Security ignore rules for secrets and build files
├── architecture_diagram.png   # Workflow visual diagram
│
├── documents/                 # Sample PDF documents
│   ├── sample.pdf
│   └── sample_policy.pdf
│
├── vector_store/              # Saved index directory
│   └── saved_index/           # Persistent FAISS index & metadata
│
└── tests/                     # Evaluation test suite
    └── test_questions.csv     # 15+ comprehensive evaluation test cases
```

---

## 6. Installation & Setup

1. **Clone / Navigate to Repository**:
   ```powershell
   cd "C:\Users\anand\OneDrive\Desktop\finla_AI(Repo)"
   ```

2. **Activate Virtual Environment**:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Requirements**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Configure `.env` File**:
   Create a `.env` file in the root directory (already git-ignored):
   ```env
   GROQ_API_KEY=gsk_your_groq_api_key_here
   ```

---

## 7. How to Run the Application

Execute the following command in PowerShell:

```powershell
streamlit run app.py
```

---

## 8. Step-by-Step User Guide

1. **Upload PDF Files**: In the left sidebar, click **Upload PDF Document(s)** and choose one or more PDFs.
2. **Adjust Parameters (Optional)**: Adjust Chunk Size (default 800), Chunk Overlap (default 120), or Top-k retrieval (default 4).
3. **Process Documents**: Click the primary **⚡ Process Documents** button in the sidebar.
4. **Ask Questions**: Type your question in the bottom chat box (e.g., *"What is the annual leave entitlement?"*).
5. **View Answer & Sources**: The assistant answers and displays an expandable accordion showing exact **Document Name**, **Page Number**, and **Similarity Score**.
6. **Unavailable Questions**: If you ask about missing topics (e.g., *"What is the company stock budget?"*), the system outputs:
   > **`I could not find this information in the uploaded documents.`**
7. **Clear Chat / Reset**: Click **🧹 Clear Chat** to clear chat history without losing the vector index, or **🗑️ Reset Documents** to clear everything.

---

## 9. Security & Responsible AI
* **API Key Safety**: `.env` is protected by `.gitignore` and never committed to GitHub.
* **Document Prompt Injection Defense**: Uploaded text is treated strictly as data. Instructions inside PDFs attempting to override system rules are ignored.
* **File Size Validation**: Rejects files exceeding 25 MB.
* **High-Stakes Disclaimer**: A prominent warning advises users to verify critical decisions against original document sources.
