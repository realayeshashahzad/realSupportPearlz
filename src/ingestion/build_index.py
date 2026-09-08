import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import settings
from src.utils.logging_setup import logger
from src.ingestion.loaders import KnowledgeBaseLoader
from src.ingestion.chunking import chunk_documents
from src.retrieval.vector_store import get_embeddings_model
from langchain_chroma import Chroma

def build_index():
    settings.validate_keys()
    logger.info("Starting Knowledge Base ingestion process...")
    
    loader = KnowledgeBaseLoader(settings.knowledge_base_dir)
    documents = loader.load_all()
    
    if not documents:
        logger.error("No documents loaded. Aborting index build.")
        return

    logger.info(f"Loaded {len(documents)} raw document sections. Splitting into chunks...")
    chunks = chunk_documents(documents)
    logger.info(f"Generated {len(chunks)} chunks using chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}.")

    persist_dir = str(settings.vector_store_dir)
    logger.info(f"Persisting vectors into Chroma database at: {persist_dir}")
    
    embeddings = get_embeddings_model()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="pearlz_kb"
    )
    
    logger.info(f"Successfully built and persisted index with {vector_store._collection.count()} chunks.")

if __name__ == "__main__":
    build_index()