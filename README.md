# 🤖 RAG Gen AI

A **Retrieval-Augmented Generation (RAG) based Generative AI application** that allows users to ask questions about documents and receive context-aware answers using information retrieved from the uploaded knowledge base.

This project demonstrates the practical implementation of a RAG pipeline, where relevant information is retrieved from documents before generating an answer using a Large Language Model (LLM).

## 🚀 Project Overview

The project is designed to build a document-based AI question-answering system.

Instead of relying only on the knowledge stored inside an LLM, the application retrieves relevant information from provided PDF documents and uses that information as context for generating the final response.

The repository contains separate backend and frontend Python files:

- `backend.py` – Handles the backend/RAG processing and document-related operations.
- `frontend.py` – Provides the user-facing interface for interacting with the application.
- `requirements.txt` – Contains the Python dependencies required for the project.
- PDF documents – Used as the knowledge sources for the RAG system.

## 🎯 Objectives

- Build a Retrieval-Augmented Generation (RAG) application
- Enable users to ask questions about PDF documents
- Retrieve relevant information from the document knowledge base
- Provide context-aware answers using Generative AI
- Understand the complete RAG workflow
- Integrate document retrieval with an LLM
- Develop a practical Generative AI application using Python

## 📚 Documents Used

The repository contains the following knowledge sources:

- **Income Tax PDF** – Used for answering questions related to income tax information.
- **Company's Leave Policy PDF** – Used for answering questions related to company leave policies.
- **Introduction to Stock Markets PDF** – Used for answering questions related to stock market fundamentals.

These documents act as the knowledge base for the RAG application.

## 🔄 RAG Workflow

```text
                 ┌──────────────────────┐
                 │     PDF Documents    │
                 │                      │
                 │ • Income Tax         │
                 │ • Company Leave      │
                 │ • Stock Markets      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  Document Processing │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Text Chunking      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │     Embeddings       │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Vector Retrieval   │
                 └──────────┬───────────┘
                            │
                User Question
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Relevant Context     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │       LLM            │
                 │  Response Generation │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    Final Answer      │
                 └──────────────────────┘
````

## ✨ Key Features

### 📄 Document-Based Question Answering

Users can ask questions related to the documents available in the knowledge base.

### 🔍 Retrieval-Augmented Generation

The application retrieves relevant document content and provides it as context to the language model before generating an answer.

### 🧠 Context-Aware Responses

The system uses retrieved information to generate responses that are relevant to the user's question and the available documents.

### 💬 Interactive AI Interface

The frontend provides an interface through which users can interact with the RAG application and ask questions.

### 📚 Multiple Knowledge Sources

The system can work with multiple PDF documents covering different topics.

## 🛠️ Technologies Used

* **Python**
* **Generative AI**
* **Large Language Models (LLMs)**
* **Retrieval-Augmented Generation (RAG)**
* **Document Processing**
* **Text Embeddings**
* **Vector Search**
* **PDF Processing**
* **Frontend & Backend Development**
* **Visual Studio Code (VS Code)**

## 📁 Project Structure

```text
RAG-gen-ai/
│
├── backend.py
├── frontend.py
├── income_tax.pdf
├── Company's leave policy.pdf
├── Introduction to Stock Markets.pdf
├── requirements.txt
│
└── README.md
```

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/vujwal4/RAG-gen-ai.git
cd RAG-gen-ai
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

Run the appropriate Python file according to the application structure:

```bash
python backend.py
```

and/or

```bash
python frontend.py
```

## 💡 Example Questions

The application can be used to ask questions such as:

### Income Tax

```text
What are the important provisions related to income tax?
```

### Company Leave Policy

```text
What is the company's leave policy?
```

### Stock Market

```text
What is a stock market?
```

```text
What are the basic concepts of stock market investing?
```

The application retrieves relevant information from the corresponding documents and uses it to generate an answer.

## 📌 Why RAG?

Traditional LLM applications rely primarily on the model's pre-trained knowledge. This can be problematic when the required information is contained in private, domain-specific, or custom documents.

RAG addresses this by combining:

```text
User Question
      ↓
Document Retrieval
      ↓
Relevant Context
      ↓
LLM
      ↓
Generated Answer
```

This approach allows an AI application to use information from a custom knowledge base while generating natural-language responses.

## 📚 Learning Outcomes

Through this project, I gained practical experience in:

* Retrieval-Augmented Generation (RAG)
* Generative AI
* Large Language Models
* Document-based question answering
* PDF document processing
* Text chunking
* Embeddings
* Vector-based retrieval
* Prompt engineering
* Backend development
* Frontend development
* Python
* Building AI-powered applications

## 💼 Skills Demonstrated

**Python | Generative AI | RAG | LLM | NLP | Embeddings | Vector Search | PDF Processing | Prompt Engineering | Backend Development | Frontend Development**

## 📌 Project Purpose

This project was developed as part of my **Generative AI and LLM portfolio** to demonstrate the practical implementation of a Retrieval-Augmented Generation system using custom PDF documents.

It showcases how domain-specific documents can be connected to a Generative AI application to build a useful document question-answering system.

## 👨‍💻 Author

**Vujwal**

GitHub: [https://github.com/vujwal4](https://github.com/vujwal4)

---
