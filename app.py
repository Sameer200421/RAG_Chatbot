import streamlit as st
from graph import rag_graph
from ingest import ingest_all_sources, create_vectorstore

st.set_page_config(page_title="RAG Chatbot", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .main {
        max-width: 800px;
        margin: 0 auto;
        background-color: #343541;
    }
    .stApp {
        background-color: #343541;
    }
    .stTextInput > div > div > input {
        border-radius: 24px;
        padding: 12px 20px;
        font-size: 1rem;
        border: 1px solid #565869;
        background-color: #40414f;
        color: #ececf1;
    }
    .stButton > button {
        border-radius: 50%;
        width: 40px;
        height: 40px;
        padding: 0;
        border: none;
        background-color: #10a37f;
        color: white;
        font-size: 1.2rem;
    }
    .stButton > button:hover {
        background-color: #0d8c6d;
    }
    h1 {
        text-align: center;
        font-weight: 600;
        margin-bottom: 3rem;
        color: #ececf1;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        background-color: #444654;
        color: #ececf1;
    }
    .chat-message strong {
        color: #ececf1;
    }
</style>
""", unsafe_allow_html=True)

st.title("Multi-Agent RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.container():
        st.markdown(f'<div class="chat-message"><strong>{message["role"]}:</strong> {message["content"]}</div>', unsafe_allow_html=True)

col1, col2 = st.columns([9, 1])

with col1:
    query = st.text_input("", placeholder="Message RAG Chatbot", label_visibility="collapsed", key="query_input")

with col2:
    submit = st.button("↑")

if submit and query:
    st.session_state.messages.append({"role": "You", "content": query})
    
    with st.spinner("Thinking..."):
        try:
            docs = ingest_all_sources(query)
            if docs:
                create_vectorstore(docs)
            result = rag_graph.invoke({"query": query})
            answer = result["answer"]
            st.session_state.messages.append({"role": "Assistant", "content": answer})
        except Exception as e:
            st.session_state.messages.append({"role": "Assistant", "content": f"Error: {e}"})
    
    st.rerun()