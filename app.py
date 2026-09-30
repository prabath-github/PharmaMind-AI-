import streamlit as st
import traceback

# Import the reusable RAG components from main.py.
# main.py must not start a terminal input() loop or rebuild the database on import.
from main import retrieve_chunks, prompt, llm

st.set_page_config(
    page_title="Pharma AI Assistant",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px;}
    .hero {padding: 1.4rem 1.6rem; border-radius: 16px;
           background: linear-gradient(120deg,#123c69,#2876a8); color: white;
           margin-bottom: 1.2rem;}
    .hero h1 {color:white; margin:0; font-size:2rem;}
    .hero p {color:#e5f1fa; margin:.35rem 0 0 0;}
    div[data-testid="stChatMessage"] {border-radius: 12px;}
    .small-muted {color:#6b7280; font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>💊 Pharma AI Assistant</h1>
  <p>Ask questions about your pharmaceutical reference book. Answers are grounded in retrieved passages.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("About")
    st.write("Retrieval-augmented pharmaceutical reference chatbot")
    st.caption("Pipeline: Chroma MMR → similarity filtering → cross-encoder reranking → Qwen")
    st.divider()
    show_sources = st.toggle("Show retrieved sources", value=True)
    show_scores = st.toggle("Show retrieval scores", value=False)
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources") and show_sources:
            with st.expander("Retrieved reference passages"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}**")
                    if show_scores:
                        st.caption(
                            f"Cosine similarity: {source['cosine_similarity']} · "
                            f"Rerank score: {source['rerank_score']}"
                        )
                    st.markdown(source["content"])
                    st.divider()

question = st.chat_input("Ask a pharmaceutical question…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.status("Searching the reference book…", expanded=True)
        try:
            results = retrieve_chunks(question)
            if not results:
                answer = (
                    "I couldn't find sufficiently relevant passages in the "
                    "reference material for this question."
                )
                sources = []
            else:
                status.update(label="Generating answer with Qwen…", state="running")
                context = "\n\n".join(
                    f"Source {i}:\n{doc.page_content}"
                    for i, doc in enumerate(results, start=1)
                )
                response = (prompt | llm).invoke({
                    "question": question,
                    "context": context,
                })
                answer = response.content.strip() if isinstance(response.content, str) else ""
                if not answer:
                    answer = "The model returned an empty answer. Please try again."
                sources = [{
                    "content": doc.page_content,
                    "cosine_similarity": doc.metadata.get("cosine_similarity"),
                    "rerank_score": doc.metadata.get("rerank_score"),
                    "metadata": doc.metadata,
                } for doc in results]

            status.update(label="Done", state="complete", expanded=False)
            st.markdown(answer)
            if sources and show_sources:
                with st.expander("Retrieved reference passages", expanded=False):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Source {i}**")
                        if show_scores:
                            st.caption(
                                f"Cosine similarity: {source['cosine_similarity']} · "
                                f"Rerank score: {source['rerank_score']}"
                            )
                        st.markdown(source["content"])
                        st.divider()
            st.session_state.messages.append({
                "role": "assistant", "content": answer, "sources": sources
            })
        except Exception as exc:
            status.update(label="An error occurred", state="error", expanded=True)
            st.error(f"{type(exc).__name__}: {exc}")
            st.caption("Check that the models and Chroma database are available.")
