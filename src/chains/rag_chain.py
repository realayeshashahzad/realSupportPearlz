from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from src.config import settings
from src.utils.logging_setup import logger
from src.retrieval.retriever import RelevanceGatedRetriever
from src.chains.schemas import RAGResponse, SourceCitation
from src.chains.prompts import CONDENSE_QUESTION_PROMPT, RAG_PROMPT

class SupportPearlzAgent:
    def __init__(self, retriever: RelevanceGatedRetriever):
        settings.validate_keys()
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            openai_api_key=settings.openai_api_key
        )
        self.structured_llm = self.llm.with_structured_output(RAGResponse)

    def _rewrite_query(self, question: str, chat_history: List[BaseMessage]) -> str:
        if not chat_history:
            return question
        
        logger.info("Rewriting query using conversation history...")
        chain = CONDENSE_QUESTION_PROMPT | self.llm
        res = chain.invoke({"question": question, "chat_history": chat_history})
        rewritten = res.content.strip()
        logger.info(f"Original Query: '{question}' -> Rewritten Query: '{rewritten}'")
        return rewritten

    def _format_context(self, docs) -> str:
        formatted = []
        for idx, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            location = doc.metadata.get("location", "N/A")
            formatted.append(f"[S{idx}] Source: {source} ({location})\nContent: {doc.page_content}")
        return "\n\n".join(formatted)

    def ask(self, question: str, chat_history: List[BaseMessage] = None) -> Dict[str, Any]:
        if chat_history is None:
            chat_history = []

        rewritten_query = self._rewrite_query(question, chat_history)
        docs, passed_gate = self.retriever.retrieve_with_scores(rewritten_query)

        if not passed_gate:
            refusal_response = RAGResponse(
                answer="I could not find relevant information in our official documentation to answer your request. Please reach out to our team at support@pearlzhome.example.",
                sources=[],
                confidence="none",
                answered=False,
                uncovered_parts="Entire query missing from knowledge base."
            )
            return {"response": refusal_response, "rewritten_query": rewritten_query, "docs": []}

        formatted_context = self._format_context(docs)
        
        prompt_val = RAG_PROMPT.format_messages(
            context=formatted_context,
            chat_history=chat_history,
            question=question
        )

        try:
            response: RAGResponse = self.structured_llm.invoke(prompt_val)
        except Exception as e:
            logger.error(f"Structured output generation error: {e}")
            response = RAGResponse(
                answer="An error occurred while generating a response. Please rephrase or contact support.",
                sources=[],
                confidence="none",
                answered=False
            )

        return {
            "response": response,
            "rewritten_query": rewritten_query,
            "docs": docs
        }
    