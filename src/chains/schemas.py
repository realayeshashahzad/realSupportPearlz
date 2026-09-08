from typing import List, Literal
from pydantic import BaseModel, Field

class SourceCitation(BaseModel):
    source_file: str = Field(description="Exact file name of the document referenced.")
    location: str = Field(description="Page number, section title, or location identifier.")

class RAGResponse(BaseModel):
    answer: str = Field(
        description="Clear, grounded answer to the customer query based exclusively on provided context."
    )
    sources: List[SourceCitation] = Field(
        default_factory=list,
        description="List of distinct source documents and locations explicitly used to form the answer."
    )
    confidence: Literal["high", "partial", "none"] = Field(
        description="High if fully answered, Partial if partially answered, None if refused/unanswerable."
    )
    answered: bool = Field(
        description="True if question was successfully answered from context; False if refused or missing data."
    )
    uncovered_parts: str = Field(
        default="",
        description="Specific parts of the user question not covered in documentation, if applicable."
    )