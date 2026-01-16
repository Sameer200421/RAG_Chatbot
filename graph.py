from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ---- LLM ----
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0
)

# ---- Embeddings + Vector DB ----
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# ---- Prompt ----
prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the question.

Context:
{context}

Question:
{question}
""")


from typing import TypedDict

# ---- State ----
class GraphState(TypedDict):
    query: str
    source: str
    context: str
    answer: str


# ---- Nodes ----
def router(state: GraphState):
    q = state["query"].lower()
    if "paper" in q or "research" in q:
        state["source"] = "arxiv"
    elif "wiki" in q or "definition" in q:
        state["source"] = "wikipedia"
    else:
        state["source"] = "rag"
    return state


def retriever_agent(state: GraphState):
    docs = retriever.invoke(state["query"])
    state["context"] = "\n\n".join(doc.page_content for doc in docs)
    return state


def answer_agent(state: GraphState):
    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    response = chain.invoke(
        {"context": state["context"], "question": state["query"]}
    )

    state["answer"] = response.content
    return state


# ---- Graph ----
graph = StateGraph(GraphState)

graph.add_node("router", router)
graph.add_node("retriever", retriever_agent)
graph.add_node("answer", answer_agent)

graph.set_entry_point("router")
graph.add_edge("router", "retriever")
graph.add_edge("retriever", "answer")
graph.set_finish_point("answer")

rag_graph = graph.compile()