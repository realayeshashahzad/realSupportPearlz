from typing import List, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.config import settings
from src.utils.logging_setup import logger

class RelevanceGatedRetriever:
    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store
        self.k = settings.retrieval_k
        self.score_threshold = settings.score_threshold

    def retrieve_with_scores(self, query: str) -> Tuple[List[Document], bool]:
        logger.info(f"Performing similarity search for query: '{query}'")
        
        results_with_scores = self.vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=self.k
        )

        filtered_docs = []
        for doc, score in results_with_scores:
            logger.info(f"Hit: {doc.metadata.get('source')} [{doc.metadata.get('location')}] - Score: {score:.4f}")
            if score >= self.score_threshold:
                doc.metadata["relevance_score"] = float(score)
                filtered_docs.append(doc)

        if not filtered_docs:
            logger.warning(f"Relevance Gate Triggered: No retrieved chunks passed threshold ({self.score_threshold}).")
            return [], False

        return filtered_docs, True