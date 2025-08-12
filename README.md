# 📄 RAG Chatbot with Groq + Ollama Embeddings

An interactive **Retrieval-Augmented Generation (RAG) Chatbot** built with **Streamlit**, powered by **Groq LLM API** for intelligent answers and **Ollama Embeddings** for semantic search over uploaded PDF documents.

## 🚀 Features

- **PDF Upload & Indexing**  
  Upload multiple PDF files, automatically extract text, chunk it, and store embeddings in **ChromaDB**.

- **Smart Retrieval & Chat**  
  Ask questions and get **context-aware answers** strictly based on your uploaded documents.

- **Quiz Generation**  
  Automatically generate **multiple-choice quizzes (MCQs)** from document context.

- **Summarization**  
  Create well-structured summaries from retrieved document chunks.

- **Custom UI & Themes**  
  - Gradient / image background support  
  - Adjustable overlay opacity  
  - Custom colors via sidebar  

- **Source-Aware Retrieval**  
  Retrieve only from selected PDFs when summarizing.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Vector Store**: [ChromaDB](https://www.trychroma.com/)
- **Embeddings**: [Ollama `nomic-embed-text:v1.5`](https://ollama.com/)
- **LLM Backend**: [Groq API](https://groq.com/)
- **PDF Parsing**: [PyPDF](https://pypi.org/project/pypdf/)
- **Text Splitting**: LangChain `RecursiveCharacterTextSplitter`
- **Env Handling**: `python-dotenv`

---

## 📂 Project Structure

