from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from src.config import settings
from src.utils.logging_setup import logger

def get_embeddings_model() -> OpenAIEmbeddings:
    settings.validate_keys()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key
    )

def load_or_create_vector_store(rebuild: bool = False) -> Chroma:
    persist_dir = str(settings.vector_store_dir)
    embeddings = get_embeddings_model()

    if Path(persist_dir).exists() and not rebuild:
        logger.info(f"Loading existing Chroma vector store from {persist_dir}")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="pearlz_kb"
        )
    
    logger.info(f"Creating new empty Chroma vector store at {persist_dir}")
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="pearlz_kb"
    )