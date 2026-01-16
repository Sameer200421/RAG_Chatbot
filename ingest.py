from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import WikipediaLoader, ArxivLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

def ingest_wikipedia(query):
    docs = WikipediaLoader(query=query, load_max_docs=2).load()
    return text_splitter.split_documents(docs)

def ingest_arxiv(query):
    docs = ArxivLoader(query=query, load_max_docs=2).load()
    return text_splitter.split_documents(docs)

def ingest_pdf(path):
    docs = PyPDFLoader(path).load()
    return text_splitter.split_documents(docs)

def ingest_web_search(query):
    search = SerpAPIWrapper()
    results = search.run(query)
    doc = Document(page_content=results, metadata={"source": "web_search", "query": query})
    return text_splitter.split_documents([doc])

def ingest_all_sources(query):
    all_docs = []
    try:
        all_docs.extend(ingest_wikipedia(query))
    except:
        pass
    try:
        all_docs.extend(ingest_arxiv(query))
    except:
        pass
    try:
        all_docs.extend(ingest_web_search(query))
    except:
        pass
    return all_docs

def create_vectorstore(docs):
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    vectorstore.add_documents(docs)
    return vectorstore