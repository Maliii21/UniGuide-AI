
"""
UniGuide-AI
RAG-Based University Policy Assistant
Alamein International University
"""

import html
import logging
import re
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

from src.conversation import ConversationManager
from src.rag_pipeline import RAGPipeline, RAGResponse
from src.query_translator import contains_arabic
from src.shared_retrieval import SharedRetrievalEngine


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "logo.png"

DOCUMENT_COUNT = 26
CHUNK_COUNT = 202


# ==========================================
# Page configuration
# ==========================================

from PIL import Image

FAVICON_PATH = LOGO_PATH.parent / "favicon.png"

st.set_page_config(
    page_title="UniGuide-AI | AIU",
    page_icon=Image.open(FAVICON_PATH) if FAVICON_PATH.is_file() else "🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# Custom styling
# ==========================================

st.markdown(
    """
    <style>
    .welcome-title {
        text-align: center;
        color: #172B4D;
        font-size: clamp(1.6rem, 4vw, 2.4rem);
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .welcome-description {
        text-align: center;
        color: #64748B;
        font-size: 1rem;
        line-height: 1.8;
    }

    .arabic-message {
    direction: rtl;
    text-align: right;
    line-height: 1.8;
    font-size: 1rem;
    white-space: pre-wrap;
    overflow-wrap: break-word;
    word-break: normal;
    unicode-bidi: isolate;
    max-width: 100%;
    padding-inline: 0.25rem;
    box-sizing: border-box;
    }

    .arabic-message .citation-id {
        direction: ltr;
        unicode-bidi: isolate;
        display: inline-block;
        white-space: nowrap;
        font-size: 0.85em;
        color: #64748B;
        max-width: 100%;
    }

    .arabic-message .arabic-bullet {
        display: block;
        position: relative;
        padding-inline-start: 1rem;
        margin-block: 0.35rem;
    }

    .arabic-message .arabic-bullet::before {
        content: "•";
        position: absolute;
        inset-inline-start: 0;
    }

    .chat-alert {
        padding: 1rem;
        border-radius: 0.6rem;
        line-height: 1.8;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }

    .alert-info {
        background-color: #EAF2FC;
        color: #1D4E89;
        border: 1px solid #D6E4F5;
    }

    .alert-warning {
        background-color: #FFF4D6;
        color: #805600;
        border: 1px solid #F0DDA0;
    }

    .alert-error {
        background-color: #FDEBEC;
        color: #A61B24;
        border: 1px solid #F4C5C8;
    }

    .app-footer {
        text-align: center;
        color: #64748B;
        font-size: 0.8rem;
        line-height: 1.8;
        padding: 1rem 0;
    }

    @media (max-width: 640px) {
    .welcome-description {
        font-size: 0.9rem;
    }

    .arabic-message {
        font-size: 0.95rem;
        line-height: 1.75;
        padding-inline: 0.15rem;
    }

    .arabic-message .citation-id {
        font-size: 0.78em;
    }

    .arabic-message .arabic-bullet {
        padding-inline-start: 0.85rem;
        margin-block: 0.2rem;
    }

    .app-footer {
        font-size: 0.75rem;
    }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# Session initialization
# ==========================================

if "conversation" not in st.session_state:
    st.session_state.conversation = ConversationManager()

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None



@st.cache_resource(show_spinner=False)
def get_shared_retrieval_engine():
    """Load retrieval resources once per server process."""

    return SharedRetrievalEngine(device="cpu")


def get_pipeline():
    """Initialize an independent pipeline for each session."""

    if st.session_state.pipeline is None:

        shared_engine = get_shared_retrieval_engine()

        st.session_state.pipeline = RAGPipeline(
            retrieval_engine=shared_engine,
            device="cpu"
        )

    return st.session_state.pipeline


# ==========================================
# Source rendering
# ==========================================

def render_sources(sources):

    if not sources:
        return

    with st.expander(
        f"📚 Sources ({len(sources)})",
        expanded=False
    ):

        for index, source in enumerate(sources, start=1):

            title = (
                source.get("document_title")
                or "University document"
            )

            page = source.get("page")
            chunk_id = source.get("chunk_id", "")
            url = source.get("source_url", "")

            st.text(f"{index}. {title}")

            st.caption(
                f"Page: {page if page is not None else 'N/A'}"
                f" | Chunk ID: {chunk_id}"
            )

            if isinstance(url, str) and url:

                parsed_url = urlparse(url)

                if (
                    parsed_url.scheme in ("http", "https")
                    and parsed_url.netloc
                ):
                    st.link_button(
                        "View original document",
                        url
                    )

            if index < len(sources):
                st.divider()


# ==========================================
# Bilingual text rendering
# ==========================================

def format_arabic_content(content: str) -> str:
    """Safely format Arabic assistant responses."""

    content = content.strip()

    # Reduce excessive blank lines.
    content = re.sub(r"\n[ \t]*\n+", "\n", content)

    # Escape untrusted HTML before formatting.
    safe_content = html.escape(content)

    # Keep citation IDs together and left-to-right.
    safe_content = re.sub(
        r"(\[AIU-\d{3}-P\d{3}-C\d{3}\])",
        r'<bdi class="citation-id" dir="ltr">\1</bdi>',
        safe_content
    )

    # Support bold Markdown text.
    safe_content = re.sub(
        r"\*\*([^\n]+?)\*\*",
        r"<strong>\1</strong>",
        safe_content
    )

    # Format simple bullet points.
    safe_content = re.sub(
        r"(?m)^[ \t]*[-*•][ \t]+([^\n]+)",
        r'<span class="arabic-bullet">\1</span>',
        safe_content
    )
    # Remove redundant line breaks around block-level bullets.
    safe_content = re.sub(
        r"</span>[ \t]*\n+(?=<span class=\"arabic-bullet\">)",
        "</span>",
        safe_content
    )

    return safe_content


def render_chat_text(content: str, is_user: bool = False):
    """Render Arabic and English chat messages."""

    if contains_arabic(content):

        safe_content = (
            html.escape(content)
            if is_user
            else format_arabic_content(content)
        )

        st.markdown(
            f'<div class="arabic-message" '
            f'dir="rtl" lang="ar">'
            f'{safe_content}</div>',
            unsafe_allow_html=True
        )

    elif is_user:
        st.text(content)

    else:
        st.markdown(content)


def render_alert(content: str, status: str):
    """Render status messages with correct text direction."""

    if not contains_arabic(content):

        if status == "insufficient_context":
            st.info(content)

        elif status == "citation_error":
            st.warning(content)

        else:
            st.error(content)

        return

    alert_classes = {
        "insufficient_context": "alert-info",
        "citation_error": "alert-warning",
        "llm_error": "alert-error"
    }

    alert_class = alert_classes.get(
        status,
        "alert-info"
    )

    safe_content = html.escape(content)

    st.markdown(
        f'<div class="chat-alert {alert_class}" '
        f'dir="rtl" lang="ar" role="alert" '
        f'style="text-align: right;">'
        f'{safe_content}</div>',
        unsafe_allow_html=True
    )


# ==========================================
# Message rendering
# ==========================================

def render_message(message):

    avatar = (
        "👤" if message.role == "user"
        else "🎓"
    )

    with st.chat_message(
        message.role,
        avatar=avatar
    ):

        if message.role == "user":

            render_chat_text(
                message.content,
                is_user=True
            )

        elif message.status in (
            "insufficient_context",
            "citation_error",
            "llm_error"
        ):

            render_alert(
                message.content,
                message.status
            )

        else:

            render_chat_text(message.content)

        if message.role == "assistant":

            render_sources(message.sources)

            if message.response_time is not None:

                st.caption(
                    f"⏱ Response time: "
                    f"{message.response_time:.2f} seconds"
                )


# ==========================================
# Question processing
# ==========================================

def process_question(question):

    conversation = st.session_state.conversation

    user_message = conversation.add_user_message(
        question
    )

    render_message(user_message)

    spinner_text = (
        "جاري البحث في وثائق الجامعة وإعداد الإجابة..."
        if contains_arabic(question)
        else "Searching university documents and generating an answer..."
    )

    with st.spinner(spinner_text):

        try:

            pipeline = get_pipeline()

            response = pipeline.ask(question)

        except Exception:

            logger.exception(
                "UniGuide-AI pipeline execution failed."
            )

            response = RAGResponse(
                question=question,
                status="llm_error",
                error_message=(
                    "The system is temporarily unable "
                    "to process your question. "
                    "Please try again later."
                )
            )

        # Defensive citation validation check.
        if response.status == "success":

            validation = response.citation_validation

            if validation is None or not validation.passed:

                response.status = "citation_error"
                response.answer = None

        assistant_message = (
            conversation.add_assistant_response(
                response
            )
        )

    render_message(assistant_message)


# ==========================================
# Sidebar
# ==========================================

with st.sidebar:

    if LOGO_PATH.is_file():
        st.image(str(LOGO_PATH), width=180)

    st.title("UniGuide-AI")

    st.caption(
        "Alamein International University"
    )

    if st.button(
        "+ New chat",
        type="primary",
        width="stretch"
    ):

        st.session_state.conversation.clear()

        st.rerun()

    st.divider()

    st.subheader("About")

    st.write(
        "An intelligent university assistant "
        "that answers student questions using "
        "university policy documents."
    )

    st.divider()

    st.subheader("Knowledge Base")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Documents", DOCUMENT_COUNT)

    with col2:
        st.metric("Chunks", CHUNK_COUNT)

    st.caption("Current indexed dataset")

    st.divider()

    st.subheader("Supported Languages")

    st.write("🇬🇧 English")
    st.write("🇪🇬 العربية")

    st.divider()

    with st.expander("System Information"):

        st.caption("Retrieval-Augmented Generation")
        st.caption("Hybrid Search + Reranking")
        st.caption("Google Gemini")
        st.caption("Automatic Query Translation")
        st.caption("Citation Validation")

    st.divider()

    st.link_button(
        "Visit AIU Website",
        "https://aiu.edu.eg/",
        width="stretch"
    )

    st.caption(
        "Academic project — not an official "
        "university service."
    )


# ==========================================
# Main interface
# ==========================================

conversation = st.session_state.conversation

selected_question = None


# ==========================================
# Welcome screen
# ==========================================

if conversation.is_empty:

    if LOGO_PATH.is_file():

        left, center, right = st.columns([2, 1, 2])

        with center:
            st.image(
                str(LOGO_PATH),
                width="stretch"
            )

    st.markdown(
        """
        <div class="welcome-title">
            Welcome to UniGuide-AI
        </div>

        <div class="welcome-description">
            Your intelligent guide to university
            policies and academic regulations.
        </div>

        <div class="welcome-description"
             dir="rtl" lang="ar">
            دليلك الذكي للوائح الأكاديمية
            وسياسات الجامعة
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric("Documents", DOCUMENT_COUNT)

    with metric2:
        st.metric("Knowledge chunks", CHUNK_COUNT)

    with metric3:
        st.metric("Languages", 2)

    st.divider()

    st.subheader("Explore common questions")

    st.caption(
        "Choose a question or ask your own "
        "in Arabic or English."
    )

    examples = [
        "Is summer registration mandatory?",
        "How can I appeal my final grade?",
        "ما هي سياسة الحضور والغياب؟",
        "كيف يمكنني طلب تقدير غير مكتمل؟"
    ]

    columns = st.columns(2)

    for index, example in enumerate(examples):

        with columns[index % 2]:

            if st.button(
                example,
                key=f"example_{index}",
                width="stretch"
            ):

                selected_question = example

else:

    logo_col, title_col = st.columns(
        [1, 7],
        vertical_alignment="center"
    )

    with logo_col:

        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=100)

    with title_col:

        st.title("UniGuide-AI")

        st.caption(
            "Your intelligent university policy assistant"
        )

    st.divider()


# ==========================================
# Display conversation history
# ==========================================

for message in conversation.get_history():

    render_message(message)


# ==========================================
# Chat input
# ==========================================

question = st.chat_input(
    "Ask a question in Arabic or English...",
    max_chars=2000
)

if selected_question is not None:

    question = selected_question

if question and question.strip():

    process_question(question.strip())

    st.rerun()


# ==========================================
# Footer and disclaimer
# ==========================================

st.divider()

st.markdown(
    """
    <div class="app-footer">
        <strong>UniGuide-AI</strong>
        · Alamein International University
        <br>
        Powered by Retrieval-Augmented Generation
        &amp; Google Gemini
        <br>
        Academic project — not an official university service.
        <br>
        Always verify important academic decisions
        with official university documents.
    </div>
    """,
    unsafe_allow_html=True
)
