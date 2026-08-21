import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import streamlit as st

from app.utils.errors import KnowledgeAssistantError, user_facing_message
from app.workflow.workflow import knowledge_assistant_graph


def render_html(html: str) -> None:
    lines = [line.strip() for line in html.strip().splitlines() if line.strip()]
    st.markdown(" ".join(lines), unsafe_allow_html=True)


st.set_page_config(
    page_title="EKA | Enterprise Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>

    /* ------------------------------
       GLOBAL
    ------------------------------ */

    .stApp {
        background-color: #F7F7FA;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }


    /* ------------------------------
       SIDEBAR
    ------------------------------ */

    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E9E9EF;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 2rem;
    }

    .logo-icon {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: linear-gradient(135deg, #6750A4, #8E7CC3);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }

    .logo-title {
        font-size: 18px;
        font-weight: 700;
        color: #202124;
        margin: 0;
    }

    .logo-subtitle {
        font-size: 11px;
        color: #8A8A98;
        margin: 0;
    }

    /* ------------------------------
       SIDEBAR TEXT — explicit colors
       (fixes "Workspace" / "System Status" headers and the three
       st.caption() lines rendering invisible: Streamlit's default
       theme text color is near-white, which vanishes against our
       white sidebar background unless explicitly overridden here)
    ------------------------------ */

    section[data-testid="stSidebar"] h3 {
        color: #24232B !important;
        font-size: 13px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] small {
        color: #6B6B78 !important;
    }

    section[data-testid="stSidebar"] p {
        color: #24232B;
    }

    /* Spinner label text — same near-invisible-default issue */
    [data-testid="stSpinner"] p {
        color: #24232B !important;
    }


    /* ------------------------------
       MAIN HEADER
    ------------------------------ */

    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 1.2rem;
        border-bottom: 1px solid #EAEAF0;
        margin-bottom: 2rem;
    }

    .header-title {
        font-size: 22px;
        font-weight: 700;
        color: #24232B;
    }

    .header-subtitle {
        font-size: 13px;
        color: #8A8A98;
        margin-top: 4px;
    }

    .status {
        background-color: #EEF9F3;
        color: #2E8B57;
        padding: 8px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }


    /* ------------------------------
       HERO SECTION
    ------------------------------ */

    .hero {
        text-align: center;
        padding-top: 4rem;
        padding-bottom: 2rem;
    }

    .hero-icon {
        font-size: 42px;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 32px;
        font-weight: 700;
        color: #24232B;
        margin-bottom: 10px;
    }

    .hero-description {
        color: #858591;
        font-size: 15px;
        max-width: 620px;
        margin: auto;
        line-height: 1.6;
    }


    /* ------------------------------
       SUGGESTION CHIPS
    ------------------------------ */

    .suggestion-title {
        text-align: center;
        color: #9A9AA5;
        font-size: 12px;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }


    /* ------------------------------
       ANSWER CARD
    ------------------------------ */

    .answer-card {
        background-color: #FFFFFF;
        border: 1px solid #E9E9EF;
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 18px rgba(30, 30, 40, 0.04);
    }

    .card-title {
        font-size: 17px;
        font-weight: 700;
        color: #292833;
        margin-bottom: 1rem;
    }

    .answer-text {
        color: #4D4C57;
        font-size: 15px;
        line-height: 1.7;
    }


    /* ------------------------------
       METRIC CARDS
    ------------------------------ */

    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E9E9EF;
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
        margin-top: 1rem;
    }

    .metric-label {
        font-size: 12px;
        color: #8A8A98;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 25px;
        font-weight: 700;
        color: #6750A4;
    }


    /* ------------------------------
       ERROR CARD
    ------------------------------ */

    .error-card {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        margin-top: 1.5rem;
        color: #991B1B;
        font-size: 14px;
        line-height: 1.6;
    }


    /* ------------------------------
       FOOTER TEXT
    ------------------------------ */

    .bottom-text {
        text-align: center;
        color: #A2A2AD;
        font-size: 11px;
        margin-top: 2rem;
        padding-bottom: 1rem;
    }


    /* ------------------------------
       BUTTONS
    ------------------------------ */

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #E4E1EC;
        background-color: #FFFFFF;
        color: #55515F;
        font-size: 13px;
        transition: 0.2s;
    }

    .stButton > button:hover {
        border-color: #6750A4;
        color: #6750A4;
    }


    /* Primary button */

    div[data-testid="stFormSubmitButton"] > button {
        background-color: #6750A4;
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        height: 44px;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #56438B;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


if "result" not in st.session_state:
    st.session_state.result = None

if "query" not in st.session_state:
    st.session_state.query = ""

if "error_message" not in st.session_state:
    st.session_state.error_message = None


with st.sidebar:

    render_html(
        """
        <div class="sidebar-logo">
            <div class="logo-icon">🧠</div>
            <div>
                <p class="logo-title">EKA</p>
                <p class="logo-subtitle">Enterprise Knowledge</p>
            </div>
        </div>
        """
    )

    if st.button(
        "＋  New Conversation",
        use_container_width=True,
    ):
        st.session_state.result = None
        st.session_state.query = ""
        st.session_state.error_message = None
        st.rerun()

    st.markdown("---")

    st.markdown("### Workspace")

    st.button(
        "📚  Knowledge Base",
        use_container_width=True,
    )

    st.button(
        "📊  Analytics",
        use_container_width=True,
    )

    st.markdown("---")

    st.markdown("### System Status")

    st.success("● System Ready")

    st.caption("LangGraph workflow active")
    st.caption("LangSmith observability enabled")
    st.caption("RAGAS evaluation enabled")

render_html(
    """
    <div class="top-header">
        <div>
            <div class="header-title">
                AI Knowledge Assistant
            </div>
            <div class="header-subtitle">
                Enterprise intelligence powered by RAG
            </div>
        </div>

        <div class="status">
            ● All systems ready
        </div>
    </div>
    """
)


if st.session_state.result is None:

    render_html(
        """
        <div class="hero">
            <div class="hero-icon">💬</div>
            <div class="hero-title">
                Ask me anything
            </div>
            <div class="hero-description">
                Search your enterprise knowledge base and get accurate,
                context-aware answers powered by AI.
            </div>
        </div>
        """
    )

    render_html(
        """
        <div class="suggestion-title">
            TRY ASKING
        </div>
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "What expenses can I claim?",
            use_container_width=True,
        ):
            st.session_state.query = "What expenses can I claim?"

    with col2:
        if st.button(
            "What is the remote work policy?",
            use_container_width=True,
        ):
            st.session_state.query = "What is the remote work policy?"

    with col3:
        if st.button(
            "What are the IT security requirements?",
            use_container_width=True,
        ):
            st.session_state.query = (
                "What are the IT security requirements?"
            )

st.markdown("<br>", unsafe_allow_html=True)

with st.form(
    key="query_form",
    clear_on_submit=False,
):

    query = st.text_input(
        "Ask a question",
        value=st.session_state.query,
        placeholder=(
            "Ask about company policies, expenses, security..."
        ),
        label_visibility="collapsed",
    )

    submit = st.form_submit_button(
        "Ask Assistant →",
        use_container_width=True,
    )

    if submit:

        if not query.strip():

            st.warning("Please enter a question.")

        else:

            st.session_state.query = query
            st.session_state.error_message = None

            initial_state = {
                "query": query,
                "retrieved_chunks": [],
                "answer": "",
                "sources": [],
                "ragas_scores": None,
                "evaluation_summary": None,
            }

            with st.spinner(
                "Searching enterprise knowledge and evaluating response..."
            ):

                try:

                    result = knowledge_assistant_graph.invoke(
                        initial_state
                    )

                    st.session_state.result = result

                except KnowledgeAssistantError as error:

                    st.session_state.result = None
                    st.session_state.error_message = user_facing_message(error)

                except Exception as error:

                    st.session_state.result = None
                    st.session_state.error_message = user_facing_message(error)


if st.session_state.error_message:

    render_html(
        f"""
        <div class="error-card">
            ⚠️ {st.session_state.error_message}
        </div>
        """
    )


if st.session_state.result:

    result = st.session_state.result

    answer = result.get(
        "answer",
        "No answer generated.",
    )

    sources = result.get(
        "sources",
        [],
    )

    ragas_scores = result.get(
        "ragas_scores",
        {},
    )

    evaluation_summary = result.get(
        "evaluation_summary",
        "No evaluation summary available.",
    )

  
    render_html(
        f"""
        <div class="answer-card">
            <div class="card-title">
                🤖 Answer
            </div>
            <div class="answer-text">
                {answer}
            </div>
        </div>
        """
    )


    if sources:

        st.markdown("### 📚 Sources")

        source_columns = st.columns(
            min(len(sources), 3)
        )

        for index, source in enumerate(sources):

            with source_columns[
                index % len(source_columns)
            ]:

                st.info(f"📄 {source}")

    if ragas_scores:

        st.markdown("### 📊 Response Evaluation")

        col1, col2 = st.columns(2)

        faithfulness = ragas_scores.get(
            "faithfulness"
        )

        answer_relevancy = ragas_scores.get(
            "answer_relevancy"
        )

        with col1:

            value = (
                f"{faithfulness:.2f}"
                if faithfulness is not None
                else "N/A"
            )

            render_html(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        Faithfulness
                    </div>
                    <div class="metric-value">
                        {value}
                    </div>
                </div>
                """
            )

        with col2:

            value = (
                f"{answer_relevancy:.2f}"
                if answer_relevancy is not None
                else "N/A"
            )

            render_html(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        Answer Relevancy
                    </div>
                    <div class="metric-value">
                        {value}
                    </div>
                </div>
                """
            )

        st.caption(
            f"Evaluation Summary: {evaluation_summary}"
        )


render_html(
    """
    <div class="bottom-text">
        Enterprise Knowledge Assistant ·
        LangGraph · RAG · MCP · RAGAS · LangSmith
    </div>
    """
)