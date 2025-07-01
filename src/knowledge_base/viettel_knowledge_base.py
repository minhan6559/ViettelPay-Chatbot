"""
ViettelPay Knowledge Base with Contextual Retrieval

This updated version:
- Uses ContextualWordProcessor for all document processing
- Integrates OpenAI for contextual enhancement
- Processes all doc/docx files from a parent folder
- Removes CSV processor dependency
"""

import os
import pickle

# import torch
from typing import List, Optional
from pathlib import Path
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import ConfigurableField
from langchain_cohere.rerank import CohereRerank

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

from src.processor.contextual_word_processor import ContextualWordProcessor
from src.processor.text_utils import VietnameseTextProcessor

# Import configuration utility
from src.utils.config import get_cohere_api_key, get_google_api_key, get_embedding_model


class ViettelKnowledgeBase:
    """ViettelPay knowledge base with contextual retrieval enhancement"""

    def __init__(self, embedding_model: str = None):
        """
        Initialize the knowledge base

        Args:
            embedding_model: Vietnamese embedding model to use
        """
        embedding_model = embedding_model or get_embedding_model()

        # Initialize Vietnamese text processor
        self.text_processor = VietnameseTextProcessor()

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        print(f"[INFO] Using device: {self.device}")

        # Initialize embeddings with GPU support and trust_remote_code
        model_kwargs = {"device": self.device, "trust_remote_code": True}

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model, model_kwargs=model_kwargs
        )

        # Initialize retrievers as None
        self.chroma_retriever = None
        self.bm25_retriever = None
        self.ensemble_retriever = None

        self.reranker = CohereRerank(
            model="rerank-v3.5",
            cohere_api_key=get_cohere_api_key(),
        )

    def build_knowledge_base(
        self,
        documents_folder: str,
        persist_dir: str = "./knowledge_base",
        reset: bool = True,
        google_api_key: Optional[str] = None,
    ) -> None:
        """
        Build knowledge base from all Word documents in a folder

        Args:
            documents_folder: Path to folder containing doc/docx files
            persist_dir: Directory to persist the knowledge base
            reset: Whether to reset existing knowledge base
            google_api_key: Google API key for contextual enhancement (optional)

        Returns:
            None. Use the search() method to perform searches.
        """

        print(
            "[INFO] Building ViettelPay knowledge base with contextual enhancement..."
        )

        # Initialize Gemini for contextual enhancement
        if google_api_key or get_google_api_key():
            api_key = google_api_key or get_google_api_key()
            print(f"[INFO] Using Gemini 2.0 Flash for contextual enhancement")

            # Initialize the contextual word processor with Gemini
            word_processor = ContextualWordProcessor(
                llm_provider="gemini", api_key=api_key
            )
        else:
            print(
                f"[WARNING] No Google API key provided. Contextual enhancement disabled."
            )
            # Initialize without LLM client
            word_processor = ContextualWordProcessor(llm_client=None)

        # Find all Word documents in the folder
        word_files = self._find_word_documents(documents_folder)

        if not word_files:
            raise ValueError(f"No Word documents found in {documents_folder}")

        print(f"[INFO] Found {len(word_files)} Word documents to process")

        # Process all documents
        all_documents = self._process_all_word_files(word_files, word_processor)
        print(f"[INFO] Total documents processed: {len(all_documents)}")

        # Create directories
        os.makedirs(persist_dir, exist_ok=True)
        chroma_dir = os.path.join(persist_dir, "chroma")
        bm25_path = os.path.join(persist_dir, "bm25_retriever.pkl")

        # Build ChromaDB retriever (uses contextualized content)
        print("[INFO] Building ChromaDB retriever with contextualized content...")
        self.chroma_retriever = self._build_chroma_retriever(
            all_documents, chroma_dir, reset
        )

        # Build BM25 retriever (uses contextualized content with Vietnamese tokenization)
        print("[INFO] Building BM25 retriever with Vietnamese tokenization...")
        self.bm25_retriever = self._build_bm25_retriever(
            all_documents, bm25_path, reset
        )

        # Create ensemble retriever with configurable top-k
        print("[INFO] Creating ensemble retriever...")
        self.ensemble_retriever = self._build_retriever(
            self.bm25_retriever, self.chroma_retriever
        )

        print("[SUCCESS] Contextual knowledge base built successfully!")
        print("[INFO] Use kb.search(query, top_k) to perform searches.")

    def load_knowledge_base(self, persist_dir: str = "./knowledge_base") -> bool:
        """
        Load existing knowledge base from disk and rebuild BM25 from ChromaDB documents

        Args:
            persist_dir: Directory where the knowledge base is stored

        Returns:
            bool: True if loaded successfully, False otherwise
        """

        print("[INFO] Loading knowledge base from disk...")

        chroma_dir = os.path.join(persist_dir, "chroma")

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
                return False

            # Extract all documents from ChromaDB to rebuild BM25
            print("[INFO] Extracting documents from ChromaDB to rebuild BM25...")
            try:
                # Get all documents and metadata from ChromaDB
                all_docs = vectorstore.get(include=["documents", "metadatas"])

                documents = all_docs["documents"]
                metadatas = all_docs["metadatas"]

                # Reconstruct Document objects
                doc_objects = []
                for i, (doc_content, metadata) in enumerate(zip(documents, metadatas)):
                    # Handle case where metadata might be None
                    if metadata is None:
                        metadata = {}

                    doc_obj = Document(page_content=doc_content, metadata=metadata)
                    doc_objects.append(doc_obj)

                print(f"[INFO] Extracted {len(doc_objects)} documents from ChromaDB")

                # Rebuild BM25 retriever using existing method
                self.bm25_retriever = self._build_bm25_retriever(
                    documents=doc_objects,
                    bm25_path=None,  # Not used anymore
                    reset=False,  # Not relevant for rebuilding
                )

            except Exception as e:
                print(f"[ERROR] Error rebuilding BM25 from ChromaDB: {e}")
                return False

            # Create ensemble retriever with configurable top-k
            self.ensemble_retriever = self._build_retriever(
                self.bm25_retriever, self.chroma_retriever
            )

            print("[SUCCESS] Knowledge base loaded successfully!")
            print("[INFO] Use kb.search(query, top_k) to perform searches.")
            return True

        except Exception as e:
            print(f"[ERROR] Error loading knowledge base: {e}")
            return False

    def search(self, query: str, top_k: int = 10) -> List[Document]:
        """
        Main search method using ensemble retriever with configurable top-k

        Args:
            query: Search query
            top_k: Number of documents to return from each retriever (default: 5)

        Returns:
            List of retrieved documents
        """
        if not self.ensemble_retriever:
            raise ValueError(
                "Knowledge base not loaded. Call build_knowledge_base() or load_knowledge_base() first."
            )

        # Build config based on top_k parameter
        config = {
            "configurable": {
                "bm25_k": top_k * 10,
                "chroma_search_kwargs": {"k": top_k * 10},
            }
        }

        results = self.ensemble_retriever.invoke(query, config=config)
        reranked_results = self.reranker.rerank(results, query, top_n=top_k)

        final_results = []
        for rerank_item in reranked_results:
            # Get the original document using the index
            original_doc = results[rerank_item["index"]]

            # Create a new document with the relevance score added to metadata
            reranked_doc = Document(
                page_content=original_doc.page_content,
                metadata={
                    **original_doc.metadata,
                    "relevance_score": rerank_item["relevance_score"],
                },
            )
            final_results.append(reranked_doc)

        return final_results

    def get_stats(self) -> dict:
        """Get statistics about the knowledge base"""
        stats = {}

        if self.chroma_retriever:
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

    def _find_word_documents(self, folder_path: str) -> List[str]:
        """
        Find all Word documents (.doc, .docx) in the given folder

        Args:
            folder_path: Path to the folder to search

        Returns:
            List of full paths to Word documents
        """
        word_files = []
        folder = Path(folder_path)

        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Search for Word documents
        for pattern in ["*.doc", "*.docx"]:
            word_files.extend(folder.glob(pattern))

        # Convert to string paths and sort for consistent processing order
        word_files = [str(f) for f in word_files]
        word_files.sort()

        print(f"[INFO] Found Word documents: {[Path(f).name for f in word_files]}")
        return word_files

    def _process_all_word_files(
        self, word_files: List[str], word_processor: ContextualWordProcessor
    ) -> List[Document]:
        """Process all Word files into unified chunks with contextual enhancement"""
        all_documents = []

        for file_path in word_files:
            try:
                print(f"[INFO] Processing: {Path(file_path).name}")
                chunks = word_processor.process_word_document(file_path)
                all_documents.extend(chunks)

                # Print processing stats for this file
                stats = word_processor.get_document_stats(chunks)
                print(
                    f"[SUCCESS] Processed {Path(file_path).name}: {len(chunks)} chunks"
                )
                print(f"  - Contextualized: {stats.get('contextualized_docs', 0)}")
                print(
                    f"  - Non-contextualized: {stats.get('non_contextualized_docs', 0)}"
                )

            except Exception as e:
                print(f"[ERROR] Error processing {Path(file_path).name}: {e}")

        return all_documents

    def _build_retriever(self, bm25_retriever, chroma_retriever):
        """
        Build ensemble retriever with configurable top-k parameters

        Args:
            bm25_retriever: BM25 retriever with configurable fields
            chroma_retriever: Chroma retriever with configurable fields

        Returns:
            EnsembleRetriever with configurable retrievers
        """
        return EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.2, 0.8],  # Slightly favor semantic search
        )

    def _build_chroma_retriever(
        self, documents: List[Document], chroma_dir: str, reset: bool
    ):
        """Build ChromaDB retriever with configurable search parameters"""

        if reset and os.path.exists(chroma_dir):
            import shutil

            shutil.rmtree(chroma_dir)
            print("[INFO] Removed existing ChromaDB for rebuild")

        # Create Chroma vectorstore (uses contextualized content)
        vectorstore = Chroma.from_documents(
            documents=documents, embedding=self.embeddings, persist_directory=chroma_dir
        )

        # Create retriever with configurable search_kwargs
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5}  # default value
        ).configurable_fields(
            search_kwargs=ConfigurableField(
                id="chroma_search_kwargs",
                name="Chroma Search Kwargs",
                description="Search kwargs for Chroma DB retriever",
            )
        )

        print(
            f"[SUCCESS] ChromaDB created with {len(documents)} contextualized documents"
        )
        return retriever

    def _build_bm25_retriever(
        self, documents: List[Document], bm25_path: str, reset: bool
    ):
        """Build BM25 retriever with Vietnamese tokenization and configurable k parameter"""

        # Note: We no longer save BM25 to pickle file to avoid Streamlit Cloud compatibility issues
        # BM25 will be rebuilt from ChromaDB documents when loading the knowledge base

        # Create BM25 retriever with Vietnamese tokenizer as preprocess_func
        print("[INFO] Using Vietnamese tokenizer for BM25 on contextualized content...")
        retriever = BM25Retriever.from_documents(
            documents=documents,
            preprocess_func=self.text_processor.bm25_tokenizer,
            k=5,  # default value
        ).configurable_fields(
            k=ConfigurableField(
                id="bm25_k",
                name="BM25 Top K",
                description="Number of documents to return from BM25",
            )
        )

        print(
            f"[SUCCESS] BM25 retriever created with {len(documents)} contextualized documents"
        )
        return retriever


def test_contextual_kb(kb: ViettelKnowledgeBase, test_queries: List[str]):
    """Test function for the contextual knowledge base"""

    print("\n[INFO] Testing Contextual Knowledge Base")
    print("=" * 60)

    for i, query in enumerate(test_queries, 1):
        print(f"\n#{i} Query: '{query}'")
        print("-" * 40)

        try:
            # Test ensemble search with configurable top-k
            results = kb.search(query, top_k=3)

            if results:
                for j, doc in enumerate(results, 1):
                    content_preview = doc.page_content[:150].replace("\n", " ")
                    doc_type = doc.metadata.get("doc_type", "unknown")
                    has_context = doc.metadata.get("has_context", False)
                    context_indicator = (
                        " [CONTEXTUAL]" if has_context else " [ORIGINAL]"
                    )
                    print(
                        f"  {j}. [{doc_type}]{context_indicator} {content_preview}..."
                    )
            else:
                print("  No results found")

        except Exception as e:
            print(f"  [ERROR] Error: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize knowledge base
    kb = ViettelKnowledgeBase(
        embedding_model="dangvantuan/vietnamese-document-embedding"
    )

    # Build knowledge base from a folder of Word documents
    documents_folder = "./viettelpay_docs"  # Folder containing .doc/.docx files

    try:
        # Build knowledge base (pass Google API key here for contextual enhancement)
        kb.build_knowledge_base(
            documents_folder,
            "./contextual_kb",
            reset=True,
            google_api_key="your-google-api-key-here",  # or None to use env variable
        )

        # Alternative: Load existing knowledge base
        # success = kb.load_knowledge_base("./contextual_kb")
        # if not success:
        #     print("[ERROR] Failed to load knowledge base")

        # Test queries
        test_queries = [
            "lỗi 606",
            "không nạp được tiền",
            "hướng dẫn nạp cước",
            "quy định hủy giao dịch",
            "mệnh giá thẻ cào",
        ]

        # Test the knowledge base
        test_contextual_kb(kb, test_queries)

        # Example of runtime configuration for different top-k values
        print(f"\n[INFO] Example of runtime configuration:")
        print("=" * 50)

        # Search with different top-k values
        sample_query = "lỗi 606"

        # Search with top_k=3
        results1 = kb.search(sample_query, top_k=3)
        print(f"Search with top_k=3: {len(results1)} total results")

        # Search with top_k=8
        results2 = kb.search(sample_query, top_k=8)
        print(f"Search with top_k=8: {len(results2)} total results")

        # Show stats
        print(f"\n[INFO] Knowledge Base Stats: {kb.get_stats()}")

    except Exception as e:
        print(f"[ERROR] Error building knowledge base: {e}")
        print("[INFO] Make sure you have:")
        print("  1. Valid Google API key")
        print("  2. Word documents in the specified folder")
        print(
            "  3. Required dependencies installed (google-generativeai, markitdown, etc.)"
        )
