import os
import base64
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv

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

# Load environment variables from .env if present
load_dotenv()

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


def _get_sanitized_groq_api_key() -> str | None:
    """Fetch Groq API key from UI/session/env and sanitize it."""
    # Prefer user-entered then session then env
    ui_key = st.session_state.get("_ui_groq_api_key", "")
    env_key = (
        os.environ.get("GROQ_API_KEY")
        or os.environ.get("GROQ_APIKEY")
        or os.environ.get("GROQ_KEY")
    )
    candidate = ui_key or st.session_state.get("groq_api_key") or env_key
    if not candidate:
        return None
    candidate = candidate.strip().strip('"').strip("'")
    return candidate or None


def _get_groq_client():
    """Create Groq client using sanitized key; returns (client, error_message)."""
    api_key = _get_sanitized_groq_api_key()
    if not api_key:
        return None, "Groq API key not found. Add it in the sidebar or .env."
    try:
        return Groq(api_key=api_key), None
    except Exception as exc:
        return None, f"Failed to initialize Groq client: {exc}"


def retrieve_and_generate_answer(
    retrieval_query: str,
    groq_client,
    groq_model_name: str,
    k: int = DEFAULT_RETRIEVAL_K,
    sources: list[str] | None = None,
    generation_instructions: str | None = None,
):
    """Retrieve relevant chunks and generate an answer using Groq.

    - retrieval_query: used for vector similarity search
    - generation_instructions: used as the user-visible task; if None, defaults to retrieval_query
    - sources: optional list of file basenames to filter retrieval on
    """
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model,
    )

    search_kwargs = {"k": k}
    if sources:
        search_kwargs["filter"] = {"source": {"$in": sources}}

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    retrieved_docs = retriever.get_relevant_documents(retrieval_query)

    if not retrieved_docs:
        return "No relevant documents found in the index.", []

    context_text = "\n\n".join([
        f"Chunk (source={doc.metadata.get('source', 'unknown')}, id={doc.metadata.get('chunk_id')}):\n{doc.page_content}"
        for doc in retrieved_docs
    ])

    system_prompt = (
        """
You are **Jarvis**, a professional study & learning assistant. You answer using only the provided context and your general reasoning; when the context is insufficient, say so and propose the minimal follow-up needed. Cite sources inline like [source_id:chunk] when available.
 
Requirements:
- Be concise, correct, and structured; prefer bullet points and short paragraphs.
- Never fabricate citations or unknown facts.
- If the user asks for a summary, deliver a faithful, well-structured summary with key points and (if relevant) timestamps.
- If asked to quiz, generate clear questions with answers based on context.
- If content appears contradictory, highlight uncertainties.
- If the user requests code, provide runnable snippets.
""".strip()
 
    )

    task_text = generation_instructions or retrieval_query
    user_prompt = f"###Context\n{context_text}\n\n###Task\n{task_text}"

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
# PARSERS & RENDERERS
# =========================
def parse_mcq_output(model_text: str) -> list[dict]:
    """Parse MCQ text in the enforced format into a structured list.

    Expected format per item:
      Q: <question>
      Options:
      A) <...>
      B) <...>
      C) <...>
      D) <...>
      Correct: <A/B/C/D>
      Explain: <text>
    """
    if not model_text:
        return []

    lines = [line.strip() for line in model_text.splitlines()]
    questions: list[dict] = []
    current: dict | None = None
    reading_options = False

    def _commit_current():
        nonlocal current
        if current is not None:
            # Ensure options dict exists
            current.setdefault("options", {})
            questions.append(current)
        current = None

    for raw in lines:
        if not raw:
            continue

        if raw.startswith("Q:"):
            _commit_current()
            current = {"question": raw[2:].strip(), "options": {}, "correct": None, "explain": ""}
            reading_options = False
            continue

        if current is None:
            continue

        if raw.lower().startswith("options"):
            reading_options = True
            continue

        if reading_options and (len(raw) > 2 and raw[1] == ")" and raw[0] in "ABCD"):
            letter = raw[0]
            text = raw[2:].strip()
            current["options"][letter] = text
            continue

        if raw.startswith("Correct:"):
            reading_options = False
            value = raw.split(":", 1)[1].strip()
            if value:
                current["correct"] = value[0].upper()
            continue

        if raw.startswith("Explain:"):
            current["explain"] = raw.split(":", 1)[1].strip()
            continue

    _commit_current()

    # Filter out incomplete entries
    cleaned: list[dict] = []
    for item in questions:
        if not item.get("question"):
            continue
        if not item.get("options"):
            continue
        cleaned.append(item)
    return cleaned


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="RAG Chatbot", layout="wide", initial_sidebar_state="expanded")
st.title("📄 RAG Chatbot")

col_upload, col_chat = st.columns([1, 2])

# ----- PDF Upload & Indexing -----

st.sidebar.header("Upload & Index PDFs")
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
groq_model_name = st.sidebar.text_input("Groq Model Name", value=GROQ_DEFAULT_MODEL)
# Secure Groq API key input (optional; overrides .env for this session)
api_key_input = st.sidebar.text_input("Groq API Key", value="", type="password", help="Override .env key for this session")
if api_key_input:
    st.session_state["_ui_groq_api_key"] = api_key_input
    st.session_state["groq_api_key"] = api_key_input

# ----- Appearance Controls -----
st.sidebar.header("Appearance")
bg_choice = st.sidebar.selectbox("Background", ["Default", "Gradient", "Image"], index=0)
primary_color = st.sidebar.color_picker("Primary color", value="#6C63FF")
accent_color = st.sidebar.color_picker("Accent color", value="#00D1B2")
overlay_opacity = st.sidebar.slider("Overlay opacity", 0.0, 0.8, 0.35, 0.05)

def _encode_file_to_base64(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None

bg_image_data_url = None
bg_image_option = None
if bg_choice == "Image":
    try:
        image_dir = "./image"
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    except Exception:
        image_files = []
    bg_image_option = st.sidebar.selectbox("Choose background image", options=image_files or ["(none)"])
    if image_files and bg_image_option in image_files:
        data_b64 = _encode_file_to_base64(os.path.join(image_dir, bg_image_option))
        if data_b64:
            bg_image_data_url = f"data:image/{bg_image_option.split('.')[-1]};base64,{data_b64}"

# Inject CSS theme
gradient_css = f"linear-gradient(135deg, {primary_color} 0%, {accent_color} 100%)"
overlay_rgba = f"rgba(0,0,0,{overlay_opacity})"
background_css = (
    f"linear-gradient({overlay_rgba}, {overlay_rgba}), url('{bg_image_data_url}') center/cover no-repeat fixed"
    if (bg_choice == "Image" and bg_image_data_url)
    else (f"linear-gradient({overlay_rgba}, {overlay_rgba}), {gradient_css}" if bg_choice == "Gradient" else "initial")
)

custom_css = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
  html, body, [data-testid="stAppViewContainer"], .stApp {{
    font-family: 'Poppins', sans-serif !important;
    background: {background_css} !important;
  }}

  /* Keep default header visible but transparent */
  [data-testid="stHeader"] {{
    background: transparent !important;
  }}

  /* Main content card look */
  .block-container {{
    padding-top: 2.75rem !important;
  }}

  /* Sidebar styling */
  [data-testid="stSidebar"] > div:first-child {{
    background: rgba(255,255,255,0.65);
    backdrop-filter: blur(8px);
  }}

  /* Tabs styling */
  button[role="tab"] {{
    border-radius: 8px !important;
    padding: 0.4rem 0.9rem !important;
    margin-right: 0.3rem !important;
    background: rgba(255,255,255,0.65) !important;
    color: #1f2937 !important;
  }}
  button[aria-selected="true"][role="tab"] {{
    background: {primary_color} !important;
    color: white !important;
    box-shadow: 0 6px 18px rgba(0,0,0,0.18);
  }}

  /* Buttons */
  .stButton > button {{
    background: {primary_color} !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1rem !important;
    box-shadow: 0 6px 16px rgba(0,0,0,0.2);
  }}
  .stButton > button:hover {{
    filter: brightness(1.05);
    transform: translateY(-1px);
  }}

  /* Expanders */
  details {{
    background: rgba(255,255,255,0.9) !important; /* light background */
    border-radius: 12px !important;
    padding: 0.4rem 0.6rem !important;
    color: #111 !important; /* dark text */
  }}
  details *, [data-testid="stExpander"] * {{
    color: #111 !important;
  }}

  /* Markdown cards used in quiz */
  .mcq-card {{
    background: rgba(255,255,255,0.85);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 10px 24px rgba(0,0,0,0.15);
  }}
  .mcq-card, .mcq-card * {{
    color: #111 !important; /* ensure readable text on white */
  }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

if st.sidebar.button("Index PDFs"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
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

# ----- Sections: Chat, Quiz, Summary -----

chat_tab, quiz_tab, summary_tab = st.tabs(["💬 Chat", "📝 Quiz", "🧾 Summary"])

with chat_tab:
    st.header("Ask a Question")
    question = st.text_area("Type your question here", height=120, key="chat_question")
    retrieval_k_chat = st.number_input(
        "Number of Chunks to Retrieve", min_value=1, max_value=20, value=DEFAULT_RETRIEVAL_K, key="chat_k"
    )

    if st.button("Get Answer", key="btn_get_answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            groq_client, err = _get_groq_client()
            if err:
                st.error(err)
            else:
                # For chat, use the user's question for both retrieval and generation
                answer, retrieved_chunks = retrieve_and_generate_answer(
                    retrieval_query=question,
                    groq_client=groq_client,
                    groq_model_name=groq_model_name,
                    k=retrieval_k_chat,
                )
                st.subheader("Answer")
                st.write(answer)

                with st.expander("Retrieved Chunks"):
                    for doc in retrieved_chunks:
                        st.markdown(
                            f"**Source:** {doc.metadata.get('source', '-')}, Chunk ID: {doc.metadata.get('chunk_id', '-') }"
                        )
                        st.write(doc.page_content[:1000] + ("..." if len(doc.page_content) > 1000 else ""))

with quiz_tab:
    st.header("Quiz Generator")
    quiz_topic = st.text_input(
        "Topic or instruction (optional)", placeholder="e.g., Create a quiz on procurement workflow"
    )
    num_questions = st.number_input("Number of questions", min_value=1, max_value=20, value=5)
    retrieval_k_quiz = st.number_input(
        "Number of Chunks to Retrieve", min_value=1, max_value=20, value=DEFAULT_RETRIEVAL_K, key="quiz_k"
    )

    if st.button("Generate Quiz", key="btn_generate_quiz"):
        groq_client, err = _get_groq_client()
        if err:
            st.error(err)
        else:
            focus = f" Focus on: {quiz_topic}." if quiz_topic.strip() else ""
            quiz_prompt = (
                f"Create a multiple-choice quiz of {num_questions} questions strictly based on the provided context."
                + focus +
                " For each question, use EXACTLY this format:\n"
                "Q: <question>\n"
                "Options:\n"
                "A) <option A>\nB) <option B>\nC) <option C>\nD) <option D>\n"
                "Correct: <A/B/C/D>\n"
                "Explain: <one-sentence rationale>\n"
                "Do not include information not present in the context."
            )
            # For quiz, retrieve on the topic but generate the formatted MCQ quiz
            answer, retrieved_chunks = retrieve_and_generate_answer(
                retrieval_query=(quiz_topic or "quiz on uploaded documents"),
                groq_client=groq_client,
                groq_model_name=groq_model_name,
                k=retrieval_k_quiz,
                generation_instructions=quiz_prompt,
            )
            st.subheader("Quiz")
            parsed = parse_mcq_output(answer)
            if not parsed:
                # Fallback to raw text if parsing fails
                st.write(answer)
            else:
                for idx, q in enumerate(parsed, start=1):
                    with st.container():
                        st.markdown(f"<div class='mcq-card'><strong>Q{idx}. {q.get('question','').strip()}</strong>", unsafe_allow_html=True)
                        options = q.get("options", {})
                        for letter in ["A", "B", "C", "D"]:
                            if letter in options:
                                st.markdown(f"- {letter}) {options[letter]}")

                        with st.expander("Show Answer"):
                            st.markdown(f"**Correct:** {q.get('correct','-')}")
                            if q.get("explain"):
                                st.markdown(f"**Explain:** {q['explain']}")
                        st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("Retrieved Chunks"):
                for doc in retrieved_chunks:
                    st.markdown(
                        f"**Source:** {doc.metadata.get('source', '-')}, Chunk ID: {doc.metadata.get('chunk_id', '-') }"
                    )
                    st.write(doc.page_content[:1000] + ("..." if len(doc.page_content) > 1000 else ""))

with summary_tab:
    st.header("Summary")
    summary_instruction = st.text_area(
        "What would you like to summarize?",
        value="Summarize the uploaded PDFs (overall).",
        height=120,
        key="summary_instruction",
    )
    retrieval_k_summary = st.number_input(
        "Number of Chunks to Retrieve", min_value=1, max_value=20, value=DEFAULT_RETRIEVAL_K, key="summary_k"
    )
    min_words = st.number_input(
        "Minimum words", min_value=100, max_value=2000, value=200, step=50, key="summary_min_words"
    )
    # Allow restricting to specific PDFs to avoid off-topic retrieval
    try:
        available_sources = sorted([
            f for f in os.listdir(UPLOAD_DIRECTORY) if f.lower().endswith(".pdf")
        ])
    except Exception:
        available_sources = []
    selected_sources = st.multiselect(
        "Limit summary to these PDFs (optional)", options=available_sources, default=[]
    )

    if st.button("Generate Summary", key="btn_generate_summary"):
        if not summary_instruction.strip():
            st.warning("Please enter what you want summarized.")
        else:
            groq_client, err = _get_groq_client()
            if err:
                st.error(err)
            else:
                generation_instructions = (
                    "Write a cohesive summary strictly based on the retrieved context from the uploaded PDFs. "
                    f"The summary must be at least {int(min_words)} words. "
                    "Organize with short headings and bullet points for key ideas when helpful. "
                    "If context seems insufficient or contradictory, state that clearly."
                )
                # Use the user's instruction as the retrieval query to steer toward the right topic (e.g., 'gen ai')
                answer, retrieved_chunks = retrieve_and_generate_answer(
                    retrieval_query=summary_instruction.strip(),
                    groq_client=groq_client,
                    groq_model_name=groq_model_name,
                    k=retrieval_k_summary,
                    sources=selected_sources if selected_sources else None,
                    generation_instructions=generation_instructions,
                )
                st.subheader("Summary Result")
                st.write(answer)

                with st.expander("Retrieved Chunks"):
                    for doc in retrieved_chunks:
                        st.markdown(
                            f"**Source:** {doc.metadata.get('source', '-')}, Chunk ID: {doc.metadata.get('chunk_id', '-') }"
                        )
                        st.write(doc.page_content[:1000] + ("..." if len(doc.page_content) > 1000 else ""))

st.caption("🔹 Using Ollama Embeddings: nomic-embed-text:v1.5")

