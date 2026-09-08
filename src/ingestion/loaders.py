import re
from pathlib import Path
from typing import List
import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from src.utils.logging_setup import logger

class KnowledgeBaseLoader:
    def __init__(self, directory: Path):
        self.directory = directory

    def _determine_doc_type(self, file_path: Path) -> str:
        name = file_path.stem.lower()
        if "manual" in name:
            return "product_manual"
        elif "warranty" in name:
            return "warranty_policy"
        elif "refund" in name or "return" in name:
            return "refund_policy"
        elif "shipping" in name:
            return "shipping_policy"
        elif "pricing" in name or "price" in name:
            return "pricing_guide"
        elif "troubleshoot" in name:
            return "troubleshooting_guide"
        elif "faq" in name:
            return "faq"
        return "general_document"

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def load_single_file(self, file_path: Path) -> List[Document]:
        documents = []
        doc_type = self._determine_doc_type(file_path)
        ext = file_path.suffix.lower()

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
                raw_docs = loader.load()
                for doc in raw_docs:
                    page_num = doc.metadata.get("page", 0) + 1
                    cleaned = self._clean_text(doc.page_content)
                    if cleaned:
                        documents.append(Document(
                            page_content=cleaned,
                            metadata={
                                "source": file_path.name,
                                "doc_type": doc_type,
                                "location": f"Page {page_num}",
                                "version": "1.0"
                            }
                        ))
            elif ext in [".docx", ".doc"]:
                loader = Docx2txtLoader(str(file_path))
                raw_docs = loader.load()
                for doc in raw_docs:
                    cleaned = self._clean_text(doc.page_content)
                    if cleaned:
                        documents.append(Document(
                            page_content=cleaned,
                            metadata={
                                "source": file_path.name,
                                "doc_type": doc_type,
                                "location": "Full Document",
                                "version": "1.0"
                            }
                        ))
            elif ext in [".md", ".txt"]:
                loader = TextLoader(str(file_path), encoding="utf-8")
                raw_docs = loader.load()
                for doc in raw_docs:
                    cleaned = self._clean_text(doc.page_content)
                    if cleaned:
                        documents.append(Document(
                            page_content=cleaned,
                            metadata={
                                "source": file_path.name,
                                "doc_type": doc_type,
                                "location": "Main Content",
                                "version": "1.0"
                            }
                        ))
            elif ext == ".csv":
                df = pd.read_csv(file_path)
                for idx, row in df.iterrows():
                    content = ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": file_path.name,
                            "doc_type": doc_type,
                            "location": f"Row {idx + 1}",
                            "version": "1.0"
                        }
                    ))
            else:
                logger.warning(f"Unsupported extension {ext} for file: {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to load file {file_path.name}: {str(e)}")

        return documents

    def load_all(self) -> List[Document]:
        all_docs = []
        files = [f for f in self.directory.rglob("*") if f.is_file()]
        logger.info(f"Found {len(files)} files in knowledge base directory: {self.directory}")

        for f in files:
            loaded = self.load_single_file(f)
            if loaded:
                all_docs.extend(loaded)
                logger.info(f"Successfully loaded {len(loaded)} document sections from {f.name}")

        return all_docs