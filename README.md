# 🧠 LangChain & RAG Streamlit Chatbot

This project is a practical implementation of Large Language Models using LangChain and Streamlit.  
It explores two different chatbot architectures:

1. A basic LangChain-powered chatbot  
2. A Retrieval-Augmented Generation (RAG) chatbot  

The purpose of this project is to understand how LLMs can be integrated into real-world applications with proper structure and secure API handling.

---

## 📌 Project Overview

This repository contains two chatbot implementations:

### 🔹 1. LangChain Chatbot (`app1.py`)

This version connects directly to a language model using LangChain and generates responses based on user input.

It demonstrates:
- LLM integration
- Prompt handling
- Streamlit UI interaction
- Basic conversational flow

---

### 🔹 2. RAG-Based Chatbot (`app.py`)

This version enhances the chatbot using Retrieval-Augmented Generation.

Instead of relying only on the model’s built-in knowledge, it:

- Loads documents
- Converts them into embeddings
- Retrieves relevant context
- Uses retrieved data to generate more accurate responses

This reduces hallucinations and improves factual reliability.

---

## 🗂 Project Structure

├── app1.py # LangChain chatbot
├── app.py # RAG chatbot
├── requirements.txt
├── README.md
└── .env # Environment variables (not pushed to GitHub)
## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Rutheeshwaran/langchain_streamlit_bot.git
cd langchain_streamlit_bot
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

(Mac/Linux)
```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variable Setup

Create a `.env` file in the root directory and add:

```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

Important:
- Never hardcode API keys inside Python files.
- Ensure `.env` is added to `.gitignore`.
- Revoke and regenerate tokens if accidentally exposed.

---

## ▶️ Running the Applications

### Run LangChain Chatbot

```bash
streamlit run app1.py
```

### Run RAG Chatbot

```bash
streamlit run app.py
```

The application will automatically open in your browser.

---

## 🧠 Concepts Covered

- LangChain framework
- LLM integration
- Prompt engineering basics
- Retrieval-Augmented Generation (RAG)
- Embeddings and vector-based retrieval
- Secure environment variable management
- Streamlit web application deployment

---

## 🔒 Security Practices

- API keys stored in environment variables
- `.env` excluded from version control
- No secrets committed to GitHub

---

## 🚀 Future Improvements

- Add conversational memory
- Integrate a persistent vector database (FAISS / Qdrant)
- Add dynamic document upload for RAG
- Improve UI design
- Deploy to cloud platforms

---

## 👨‍💻 Author

Rutheeshwaran Prabakaran  
Final Year – Artificial Intelligence & Data Science
