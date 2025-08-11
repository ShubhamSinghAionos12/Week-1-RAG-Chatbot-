import os
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# =========================
# CONFIGURATION
# =========================
PERSIST_DIRECTORY = "./chroma_db"
UPLOAD_DIRECTORY = "./uploaded_pdfs"
COLLECTION_NAME = "RAG"
GROQ_DEFAULT_MODEL = "gemma2-9b-it"
DEFAULT_RETRIEVAL_K = 4
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:v1.5"

# Create necessary directories
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# Initialize embedding model
embedding_model = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

# =========================
# HELPER FUNCTIONS
# =========================
def save_uploaded_pdf(uploaded_file):
    """Save uploaded PDF to the upload directory."""
    file_path = os.path.join(UPLOAD_DIRECTORY, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    reader = PdfReader(file_path)
    extracted_text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            extracted_text.append(page_text)
    return "\n".join(extracted_text)


def split_text_into_chunks(text, chunk_size=512, overlap=50):
    """Split long text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)


def create_or_update_vectorstore(text_chunks, source_file_name):
    """Create a new Chroma vectorstore or update an existing one."""
    documents = [
        Document(page_content=chunk, metadata={"source": source_file_name, "chunk_id": i})
        for i, chunk in enumerate(text_chunks)
    ]

    if not os.path.exists(PERSIST_DIRECTORY) or len(os.listdir(PERSIST_DIRECTORY)) == 0:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIRECTORY
        )
    else:
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_model
        )
        vectorstore.add_documents(documents)

    vectorstore.persist()
    return vectorstore


def retrieve_and_generate_answer(query, groq_client, groq_model_name, k=DEFAULT_RETRIEVAL_K):
    """Retrieve relevant chunks and generate an answer using Groq."""
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    retrieved_docs = retriever.get_relevant_documents(query)

    if not retrieved_docs:
        return "No relevant documents found in the index.", []

    context_text = "\n\n".join([
        f"Chunk (source={doc.metadata.get('source', 'unknown')}, id={doc.metadata.get('chunk_id')}):\n{doc.page_content}"
        for doc in retrieved_docs
    ])

    system_prompt = (
        """
You are **Athena**, a professional study & learning assistant. You answer using only the provided context and your general reasoning; when the context is insufficient, say so and propose the minimal follow-up needed. Cite sources inline like [source_id:chunk] when available.
 
Requirements:
- Be concise, correct, and structured; prefer bullet points and short paragraphs.
- Never fabricate citations or unknown facts.
- If the user asks for a summary, deliver a faithful, well-structured summary with key points and (if relevant) timestamps.
- If asked to quiz, generate clear questions with answers based on context.
- If content appears contradictory, highlight uncertainties.
- If the user requests code, provide runnable snippets.
""".strip()
 
    )

    user_prompt = f"###Context\n{context_text}\n\n###Question\n{query}"

    try:
        response = groq_client.chat.completions.create(
            model=groq_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Error calling Groq API: {e}"

    return answer, retrieved_docs


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 RAG Chatbot")

col_upload, col_chat = st.columns([1, 2])

# ----- PDF Upload & Indexing -----

st.sidebar.header("Upload & Index PDFs")
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
groq_model_name = st.sidebar.text_input("Groq Model Name", value=GROQ_DEFAULT_MODEL)
groq_api_key = os.environ.get("GROQ_API_KEY")

if st.sidebar.button("Index PDFs"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    elif not groq_api_key:
        st.error("Groq API key not found.")
    else:
        groq_client = Groq(api_key=groq_api_key)
        with st.spinner("Indexing PDFs..."):
            for uploaded_file in uploaded_files:
                pdf_path = save_uploaded_pdf(uploaded_file)
                pdf_text = extract_text_from_pdf(pdf_path)
                if not pdf_text.strip():
                    st.warning(f"No text found in {uploaded_file.name}, skipping...")
                    continue
                    chunks = split_text_into_chunks(pdf_text)
                    create_or_update_vectorstore(chunks, os.path.basename(pdf_path))
            st.success("✅ PDFs indexed successfully!")

# ----- Question & Answer -----

st.header("Ask a Question")
question = st.text_area("Type your question here", height=120)
retrieval_k = st.number_input("Number of Chunks to Retrieve", min_value=1, max_value=20, value=DEFAULT_RETRIEVAL_K)

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif not groq_api_key:
        st.error("Groq API key not found.")
    else:
        groq_client = Groq(api_key=groq_api_key)
        answer, retrieved_chunks = retrieve_and_generate_answer(question, groq_client, groq_model_name, k=retrieval_k)
        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved Chunks"):
            for doc in retrieved_chunks:
                st.markdown(f"**Source:** {doc.metadata.get('source', '-')}, Chunk ID: {doc.metadata.get('chunk_id', '-')}")
                st.write(doc.page_content[:1000] + ("..." if len(doc.page_content) > 1000 else ""))

st.caption("🔹 Using Ollama Embeddings: nomic-embed-text:v1.5")

