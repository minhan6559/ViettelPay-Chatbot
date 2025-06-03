"""
ViettelPay Knowledge Base using LangChain components

This version uses:
- LangChain Documents throughout the pipeline
- Same content for both ChromaDB and BM25 (no enhanced versions)
- LangChain's Chroma and BM25Retriever
- EnsembleRetriever for fusion
- Pickle for BM25 persistence
- Vietnamese text preprocessing for BM25
"""

import os
import pickle
import torch
from typing import Dict, List, Optional
from pathlib import Path

from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Use newest import paths for langchain
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# Use the new HuggingFaceEmbeddings from langchain-huggingface
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from src.processor.csv_processor import CSVProcessor
from src.processor.word_processor import WordDocumentProcessor
from src.processor.text_utils import VietnameseTextProcessor


class ViettelKnowledgeBase:
    """ViettelPay knowledge base using LangChain components"""

    def __init__(
        self, embedding_model: str = "dangvantuan/vietnamese-document-embedding"
    ):
        self.embedding_model = embedding_model

        # Initialize Vietnamese text processor
        self.text_processor = VietnameseTextProcessor()

        # Detect GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {self.device}")

        # Initialize embeddings with GPU support and trust_remote_code
        model_kwargs = {"device": self.device, "trust_remote_code": True}

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model, model_kwargs=model_kwargs
        )

        self.csv_processor = CSVProcessor()
        self.word_processor = WordDocumentProcessor()

        # Initialize retrievers as None
        self.chroma_retriever = None
        self.bm25_retriever = None
        self.ensemble_retriever = None

    def build_knowledge_base(
        self,
        file_paths: Dict[str, str],
        persist_dir: str = "./knowledge_base",
        reset: bool = True,
    ) -> EnsembleRetriever:
        """Build knowledge base with both retrievers and ensemble"""

        print("[INFO] Building ViettelPay knowledge base...")

        # Process all documents
        all_documents = self._process_all_files(file_paths)
        print(f"[INFO] Total documents processed: {len(all_documents)}")

        # Create directories
        os.makedirs(persist_dir, exist_ok=True)
        chroma_dir = os.path.join(persist_dir, "chroma")
        bm25_path = os.path.join(persist_dir, "bm25_retriever.pkl")

        # Build ChromaDB retriever (uses original text)
        print("[INFO] Building ChromaDB retriever...")
        self.chroma_retriever = self._build_chroma_retriever(
            all_documents, chroma_dir, reset
        )

        # Build BM25 retriever (uses Vietnamese tokenizer)
        print("[INFO] Building BM25 retriever with Vietnamese tokenization...")
        self.bm25_retriever = self._build_bm25_retriever(
            all_documents, bm25_path, reset
        )

        # Create ensemble retriever
        print("[INFO] Creating ensemble retriever...")
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.chroma_retriever],
            weights=[0.4, 0.6],  # Slightly favor semantic search
        )

        print("[SUCCESS] Knowledge base built successfully!")
        return self.ensemble_retriever

    def load_knowledge_base(
        self, persist_dir: str = "./knowledge_base"
    ) -> Optional[EnsembleRetriever]:
        """Load existing knowledge base from disk"""

        print("[INFO] Loading knowledge base from disk...")

        chroma_dir = os.path.join(persist_dir, "chroma")
        bm25_path = os.path.join(persist_dir, "bm25_retriever.pkl")

        try:
            # Load ChromaDB
            if os.path.exists(chroma_dir):
                vectorstore = Chroma(
                    persist_directory=chroma_dir, embedding_function=self.embeddings
                )
                self.chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                print("[SUCCESS] ChromaDB loaded")
            else:
                print("[ERROR] ChromaDB not found")
                return None

            # Load BM25
            if os.path.exists(bm25_path):
                with open(bm25_path, "rb") as f:
                    self.bm25_retriever = pickle.load(f)
                print("[SUCCESS] BM25 retriever loaded")
            else:
                print("[ERROR] BM25 retriever not found")
                return None

            # Create ensemble retriever
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.chroma_retriever],
                weights=[0.4, 0.6],
            )

            print("[SUCCESS] Knowledge base loaded successfully!")
            return self.ensemble_retriever

        except Exception as e:
            print(f"[ERROR] Error loading knowledge base: {e}")
            return None

    def search(self, query: str, k: int = 5) -> List[Document]:
        """Main search method using ensemble retriever"""
        if not self.ensemble_retriever:
            raise ValueError(
                "Knowledge base not loaded. Call build_knowledge_base() or load_knowledge_base() first."
            )

        # Use newer .invoke() method instead of get_relevant_documents()
        results = self.ensemble_retriever.invoke(query)
        return results[:k]

    def search_bm25_only(self, query: str, k: int = 5) -> List[Document]:
        """Search using BM25 only (good for exact matches like error codes)"""
        if not self.bm25_retriever:
            raise ValueError("BM25 retriever not loaded.")

        # Use .invoke() instead of get_relevant_documents()
        results = self.bm25_retriever.invoke(query)
        return results[:k]

    def search_semantic_only(self, query: str, k: int = 5) -> List[Document]:
        """Search using ChromaDB only (good for semantic queries)"""
        if not self.chroma_retriever:
            raise ValueError("ChromaDB retriever not loaded.")

        # Use .invoke() instead of get_relevant_documents()
        results = self.chroma_retriever.invoke(query)
        return results[:k]

    # Specialized search methods
    def search_by_error_code(self, error_code: str, k: int = 3) -> List[Document]:
        """Specialized search for error codes"""
        query = f"lỗi {error_code} error {error_code} mã lỗi {error_code}"
        return self.search_bm25_only(query, k)

    def search_procedures(self, query: str, k: int = 3) -> List[Document]:
        """Specialized search for step-by-step procedures"""
        enhanced_query = f"hướng dẫn {query} guide {query} bước {query}"
        return self.search(enhanced_query, k)

    def search_definitions(self, term: str, k: int = 2) -> List[Document]:
        """Specialized search for term definitions"""
        enhanced_query = f"định nghĩa {term} definition {term} nghĩa là {term}"
        return self.search(enhanced_query, k)

    def get_stats(self) -> dict:
        """Get statistics about the knowledge base"""
        stats = {}

        if self.chroma_retriever:
            # Try to get count from Chroma vectorstore
            try:
                vectorstore = self.chroma_retriever.vectorstore
                collection = vectorstore._collection
                stats["chroma_documents"] = collection.count()
            except:
                stats["chroma_documents"] = "Unknown"

        if self.bm25_retriever:
            try:
                stats["bm25_documents"] = len(self.bm25_retriever.docs)
            except:
                stats["bm25_documents"] = "Unknown"

        stats["ensemble_available"] = self.ensemble_retriever is not None
        stats["device"] = self.device
        stats["vietnamese_tokenizer"] = "Vietnamese BM25 tokenizer (underthesea)"

        return stats

    def _process_all_files(self, file_paths: Dict[str, str]) -> List[Document]:
        """Process all CSV and Word files into unified chunks"""
        all_documents = []

        # Define CSV processors mapping
        csv_processors = {
            "definitions": self.csv_processor.process_definitions,
            "error_handling": self.csv_processor.process_error_handling,
            "payment_guide": self.csv_processor.process_payment_guide,
            "error_codes": self.csv_processor.process_error_codes,
            "cancellation_rules": self.csv_processor.process_cancellation_rules,
            "denominations": self.csv_processor.process_denominations,
        }

        # Process CSV files
        for file_type, processor_func in csv_processors.items():
            if file_type in file_paths and os.path.exists(file_paths[file_type]):
                try:
                    chunks = processor_func(file_paths[file_type])
                    all_documents.extend(chunks)
                    print(f"[SUCCESS] Processed {file_type}: {len(chunks)} chunks")
                except Exception as e:
                    print(f"[ERROR] Error processing {file_type}: {e}")

        # Process Word document
        if "word_document" in file_paths and os.path.exists(
            file_paths["word_document"]
        ):
            try:
                word_chunks = self.word_processor.process_word_document(
                    file_paths["word_document"]
                )
                all_documents.extend(word_chunks)
                print(f"[SUCCESS] Processed Word document: {len(word_chunks)} chunks")
            except Exception as e:
                print(f"[ERROR] Error processing Word document: {e}")

        return all_documents

    def _build_chroma_retriever(
        self, documents: List[Document], chroma_dir: str, reset: bool
    ):
        """Build ChromaDB retriever"""

        if reset and os.path.exists(chroma_dir):
            import shutil

            shutil.rmtree(chroma_dir)
            print("[INFO] Removed existing ChromaDB for rebuild")

        # Create Chroma vectorstore (uses original text)
        vectorstore = Chroma.from_documents(
            documents=documents, embedding=self.embeddings, persist_directory=chroma_dir
        )

        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        print(f"[SUCCESS] ChromaDB created with {len(documents)} documents")
        return retriever

    def _build_bm25_retriever(
        self, documents: List[Document], bm25_path: str, reset: bool
    ):
        """Build BM25 retriever with Vietnamese tokenization and save with pickle"""

        if reset and os.path.exists(bm25_path):
            os.remove(bm25_path)
            print("[INFO] Removed existing BM25 retriever for rebuild")

        # Create BM25 retriever with Vietnamese tokenizer as preprocess_func
        print("[INFO] Using Vietnamese tokenizer for BM25...")
        retriever = BM25Retriever.from_documents(
            documents=documents,
            preprocess_func=self.text_processor.bm25_tokenizer,  # Use existing tokenizer
        )
        retriever.k = 5  # Set default k

        # Save with pickle
        with open(bm25_path, "wb") as f:
            pickle.dump(retriever, f)

        print(
            f"[SUCCESS] BM25 retriever created and saved with {len(documents)} documents"
        )
        print(f"[INFO] Vietnamese tokenization applied via preprocess_func")
        return retriever


# Legacy name for backward compatibility if needed
SimplifiedViettelKnowledgeBase = ViettelKnowledgeBase


def test_simplified_kb(kb: ViettelKnowledgeBase, test_queries: List[str]):
    """Simple test function for the knowledge base"""

    print("\n[INFO] Testing Knowledge Base")
    print("=" * 50)

    for i, query in enumerate(test_queries, 1):
        print(f"\n#{i} Query: '{query}'")
        print("-" * 30)

        try:
            # Test ensemble search
            results = kb.search(query, k=3)

            if results:
                for j, doc in enumerate(results, 1):
                    content_preview = doc.page_content[:100].replace("\n", " ")
                    doc_type = doc.metadata.get("doc_type", "unknown")
                    print(f"  {j}. [{doc_type}] {content_preview}...")
            else:
                print("  No results found")

        except Exception as e:
            print(f"  [ERROR] Error: {e}")


# Example usage
if __name__ == "__main__":
    # File paths setup
    data_dir = "./viettelpay_docs/processed"
    file_paths = {
        "definitions": os.path.join(data_dir, "dinh_nghia.csv"),
        "error_handling": os.path.join(data_dir, "huong_dan_xu_ly_loi.csv"),
        "payment_guide": os.path.join(data_dir, "huong_dan_thanh_toan.csv"),
        "error_codes": os.path.join(data_dir, "bang_ma_loi.csv"),
        "cancellation_rules": os.path.join(data_dir, "quy_dinh_huy_giao_dich.csv"),
        "denominations": os.path.join(data_dir, "menh_gia.csv"),
        "word_document": os.path.join(
            data_dir, "nghiep_vu_thanh_toan_cuoc_vien_thong.docx"
        ),
    }

    # Initialize knowledge base
    kb = ViettelKnowledgeBase()

    # Build knowledge base
    ensemble_retriever = kb.build_knowledge_base(
        file_paths, "./simplified_kb", reset=True
    )

    # Test queries
    test_queries = [
        "lỗi 606",
        "không nạp được tiền",
        "hướng dẫn nạp cước",
        "quy định hủy giao dịch",
    ]

    # Test the knowledge base
    test_simplified_kb(kb, test_queries)

    # Show stats
    print(f"\n[INFO] Knowledge Base Stats: {kb.get_stats()}")
