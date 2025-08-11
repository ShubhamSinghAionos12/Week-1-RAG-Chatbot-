# 📄 RAG Chatbot with Groq & Ollama Embeddings

An interactive **Retrieval-Augmented Generation (RAG) Chatbot** built with **Streamlit**, powered by:
- **Ollama Embeddings** (`nomic-embed-text:v1.5`)
- **Groq LLM API** (default: `gemma2-9b-it`)
- **ChromaDB** for vector storage

This chatbot indexes your **PDF documents** and allows you to query them with natural language.  
It retrieves the most relevant chunks and uses the Groq API to generate concise, context-based answers.

---

## 🚀 Features

- 📤 **Upload multiple PDFs** for indexing
- 🔍 **Retrieval-Augmented Generation** using ChromaDB
- 🤖 **Ollama embeddings** for semantic search
- ⚡ **Groq API** for fast and accurate answers
- 📑 **Source citations** for retrieved chunks
- 🎯 **Adjustable retrieval size** (`k` chunks)
- 🖥 **Streamlit UI** for easy interaction

---

## 🛠 Tech Stack

- **Frontend/UI:** [Streamlit](https://streamlit.io)
- **Vector Database:** [Chroma](https://www.trychroma.com/)
- **Embeddings:** [Ollama Embeddings](https://ollama.ai)
- **LLM API:** [Groq](https://groq.com/)
- **PDF Parsing:** [PyPDF](https://pypdf.readthedocs.io/)
- **Text Splitting:** LangChain’s `RecursiveCharacterTextSplitter`

---

## 📂 Project Structure

