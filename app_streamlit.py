import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.config import settings
from src.retrieval.vector_store import load_or_create_vector_store
from src.retrieval.retriever import RelevanceGatedRetriever
from src.chains.rag_chain import SupportPearlzAgent

st.set_page_config(
    page_title="SupportPearlz Knowledge Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🛡️ SupportPearlz Customer Support Agent")
st.caption("Grounded AI Assistant for Pearlz Home Systems")

@st.cache_resource
def init_agent():
    vector_store = load_or_create_vector_store(rebuild=False)
    retriever = RelevanceGatedRetriever(vector_store)
    return SupportPearlzAgent(retriever)

try:
    agent = init_agent()
except Exception as e:
    st.error(f"Initialization Failed: {str(e)}")
    st.stop()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration Overview")
    st.text(f"LLM Model: {settings.llm_model}")
    st.text(f"Embeddings: {settings.embedding_model}")
    st.text(f"Top-K Chunks: {settings.retrieval_k}")
    st.text(f"Score Threshold: {settings.score_threshold}")
    
    st.markdown("---")
    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.langchain_history = []
        st.rerun()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "langchain_history" not in st.session_state:
    st.session_state.langchain_history = []

# Render Existing Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "rewritten" in msg and msg["rewritten"]:
            st.caption(f"🔍 *Rewritten Query:* `{msg['rewritten']}`")
        if "confidence" in msg:
            conf = msg["confidence"]
            color = "green" if conf == "high" else "orange" if conf == "partial" else "red"
            st.markdown(f":{color}[**Confidence: {conf.upper()}**]")
        if "sources" in msg and msg["sources"]:
            with st.expander("📌 Source Citations"):
                for src in msg["sources"]:
                    st.write(f"- **{src.source_file}** ({src.location})")

# User Input Handling
if user_query := st.chat_input("Ask about Pearlz products, warranty, or returns..."):
    # Render user query
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Process via Agent
    with st.chat_message("assistant"):
        with st.spinner("Searching KB and generating grounded response..."):
            res_dict = agent.ask(user_query, st.session_state.langchain_history)
            rag_res = res_dict["response"]
            rewritten = res_dict["rewritten_query"]

            st.markdown(rag_res.answer)
            st.caption(f"🔍 *Rewritten Query:* `{rewritten}`")
            
            conf = rag_res.confidence
            color = "green" if conf == "high" else "orange" if conf == "partial" else "red"
            st.markdown(f":{color}[**Confidence: {conf.upper()}**]")

            if rag_res.sources:
                with st.expander("📌 Source Citations"):
                    for src in rag_res.sources:
                        st.write(f"- **{src.source_file}** ({src.location})")

    # Update Session State
    st.session_state.messages.append({
        "role": "assistant",
        "content": rag_res.answer,
        "rewritten": rewritten,
        "confidence": rag_res.confidence,
        "sources": rag_res.sources
    })
    
    st.session_state.langchain_history.append(HumanMessage(content=user_query))
    st.session_state.langchain_history.append(AIMessage(content=rag_res.answer))