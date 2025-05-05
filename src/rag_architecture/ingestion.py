import os
import re
import unicodedata
import logging
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_transformers.openai_functions import (
    create_metadata_tagger,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pdfplumber.utils.text import LIGATURES as PDF_LIGATURES

# ---------------------------------------------------------------------------- #
# Load environment variables
# ---------------------------------------------------------------------------- #
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_ATLAS_URI")
DB_NAME = os.getenv("MONGODB_DB_NAME")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME")

# ---------------------------------------------------------------------------- #
# Set up MongoDB client
# ---------------------------------------------------------------------------- #
client = MongoClient(MONGODB_URI)
db = client.get_database(DB_NAME)
vector_collection = db[COLLECTION_NAME]

# ---------------------------------------------------------------------------- #
# Logging setup
# ---------------------------------------------------------------------------- #
LOG_PREVIEW_CHARS = 120
MAX_LOG_PAGES_TO_LOG = 2
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------- #
# Ligature replacements for cleaning PDF artifacts
# ---------------------------------------------------------------------------- #
EXTRA_LIGATURES = {
    "ﬅ": "ft",
    "ﬆ": "st",
}
LIGATURES: dict[str, str] = {**PDF_LIGATURES, **EXTRA_LIGATURES}


# ---------------------------------------------------------------------------- #
# Text cleaning function
# ---------------------------------------------------------------------------- #
def clean_text(raw: str) -> str:
    """Normalize and clean raw PDF text.

    Replaces ligatures, fixes hyphenation, removes newlines, and normalizes punctuation spacing.

    Args:
        raw (str): The raw input text from a PDF page.

    Returns:
        str: A cleaned and normalized version of the input text.
    """
    text = unicodedata.normalize("NFKC", raw)
    for lig, repl in LIGATURES.items():
        text = text.replace(lig, repl)
    text = re.sub(r"(?<=\w)-\s*\n\s*", "", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = " ".join(text.split())
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[.,])(?=\S)", " ", text)
    text = re.sub(r"\s*\b\d{1,3}\s*$", "", text)
    return text.strip()


# ---------------------------------------------------------------------------- #
# Load and clean PDF pages
# ---------------------------------------------------------------------------- #
def load_and_clean(file_path: str):
    """Load a PDF and clean its contents.

    Extracts pages using LangChain's PyPDFLoader, cleans each page using `clean_text`,
    and returns only those with significant content.

    Args:
        file_path (str): Path to the PDF file to be processed.

    Returns:
        list: A list of cleaned LangChain Document objects.
    """
    pages = PyPDFLoader(file_path).load()
    cleaned_pages = []
    for idx, page in enumerate(pages):
        raw = page.page_content or ""
        if idx < MAX_LOG_PAGES_TO_LOG:
            logger.info(f"[Raw p{idx}] {raw[:LOG_PREVIEW_CHARS]!r}")
        cleaned = clean_text(raw)
        if idx < MAX_LOG_PAGES_TO_LOG:
            logger.info(f"[Cleaned p{idx}] {cleaned[:LOG_PREVIEW_CHARS]!r}")
        if len(cleaned.split()) > 20:
            page.page_content = cleaned
            cleaned_pages.append(page)
    return cleaned_pages


# ---------------------------------------------------------------------------- #
# Tag metadata using OpenAI and LangChain's transformer
# ---------------------------------------------------------------------------- #
def tag_metadata(docs):
    """Enrich documents with metadata using an LLM.

    Calls OpenAI's GPT model to tag documents with title, keywords, and code presence.

    Args:
        docs (list): A list of LangChain Document objects.

    Returns:
        list: The same documents enriched with metadata.
    """
    schema = {
        "properties": {
            "title": {"type": "string"},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "hasCode": {"type": "boolean"},
        },
        "required": ["title", "keywords", "hasCode"],
    }
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model="gpt-3.5-turbo",
        temperature=0,
    )
    transformer = create_metadata_tagger(metadata_schema=schema, llm=llm)
    transformed = transformer.transform_documents(docs)
    for idx, doc in enumerate(transformed[:MAX_LOG_PAGES_TO_LOG]):
        logger.info(f"[Tagged p{idx} metadata] {doc.metadata}")
    return transformed


# ---------------------------------------------------------------------------- #
# Chunk documents into vector-friendly pieces
# ---------------------------------------------------------------------------- #
def chunk_docs(docs):
    """Split documents into overlapping text chunks.

    Uses LangChain's RecursiveCharacterTextSplitter for chunking.

    Args:
        docs (list): A list of LangChain Document objects.

    Returns:
        list: A list of text chunks suitable for vector embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
    )
    return splitter.split_documents(docs)


# ---------------------------------------------------------------------------- #
# Store vector embeddings into MongoDB Atlas
# ---------------------------------------------------------------------------- #
def store_embeddings(chunks):
    """Generate and store embeddings in MongoDB Atlas vector store.

    Converts chunks into embeddings using OpenAI and saves them into the configured MongoDB collection.

    Args:
        chunks (list): A list of text chunks to be embedded and stored.

    Returns:
        None
    """
    embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = MongoDBAtlasVectorSearch.from_documents(
        chunks,
        embeddings_model,
        collection=vector_collection,
    )
    logger.info("✅  Stored embeddings in MongoDB Atlas vector search collection")
