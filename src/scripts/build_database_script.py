"""
ViettelPay Knowledge Base Management Script

This script uses the new ContextualWordProcessor with:
- Automated processing of Word documents (.doc/.docx) from a folder
- Contextual enhancement using OpenAI API (optional)
- LangChain EnsembleRetriever for hybrid search
- ChromaDB for semantic search and BM25 for keyword search

Usage:
    python build_database_script.py ingest --documents-folder ./viettelpay_docs
    python build_database_script.py test --query "lỗi 606"
    python build_database_script.py test --interactive
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add the project root to Python path so we can import from src
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge_base.viettel_knowledge_base import ViettelKnowledgeBase

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def validate_documents_folder(documents_folder: str) -> bool:
    """Validate that documents folder exists and contains Word documents"""

    if not os.path.exists(documents_folder):
        print(f"[ERROR] Documents folder not found: {documents_folder}")
        return False

    # Check for Word documents
    folder = Path(documents_folder)
    word_files = []
    for pattern in ["*.doc", "*.docx"]:
        word_files.extend(folder.glob(pattern))

    if not word_files:
        print(f"[ERROR] No Word documents (.doc/.docx) found in: {documents_folder}")
        return False

    print(f"[SUCCESS] Found {len(word_files)} Word documents in {documents_folder}")
    for word_file in word_files:
        print(f"   - {word_file.name}")

    return True


def ingest_documents(args):
    """Ingest documents and build knowledge base"""

    print("=" * 60)
    print("[INFO] INGESTING DOCUMENTS AND BUILDING KNOWLEDGE BASE")
    print("=" * 60)

    # Validate documents folder exists and contains Word documents
    if not validate_documents_folder(args.documents_folder):
        sys.exit(1)

    # Build knowledge base with OpenAI API key support
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        print("[INFO] Using OpenAI API key for contextual enhancement")
    else:
        print("[WARNING] No OpenAI API key found. Contextual enhancement disabled.")

    kb = ViettelKnowledgeBase(
        embedding_model=args.embedding_model, openai_api_key=openai_api_key
    )

    try:
        # Create persist directory from chroma_dir
        persist_dir = os.path.dirname(args.chroma_dir) or "./knowledge_base"

        # Build knowledge base using the new API
        kb.build_knowledge_base(
            documents_folder=args.documents_folder,
            persist_dir=persist_dir,
            reset=args.reset,
        )

        # Show final statistics
        print("\n[INFO] Knowledge Base Statistics:")
        stats = kb.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")

        print(f"\n[SUCCESS] Knowledge base saved successfully to {persist_dir}!")

        return True

    except Exception as e:
        print(f"[ERROR] Error during ingestion: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_retrieval(args):
    """Test retrieval on existing knowledge base"""

    print("=" * 60)
    print("[INFO] TESTING KNOWLEDGE BASE RETRIEVAL")
    print("=" * 60)

    # Load knowledge base
    kb = ViettelKnowledgeBase(embedding_model=args.embedding_model)

    # Create persist directory from chroma_dir
    persist_dir = os.path.dirname(args.chroma_dir) or "./knowledge_base"

    # Load knowledge base
    success = kb.load_knowledge_base(persist_dir=persist_dir)

    if not success:
        print("[ERROR] Failed to load knowledge base. Run 'ingest' first.")
        sys.exit(1)

    # Show knowledge base stats
    print("\n[INFO] Knowledge Base Statistics:")
    stats = kb.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    if args.interactive:
        # Interactive testing mode
        run_interactive_tests(kb)
    elif args.query:
        # Single query testing
        test_single_query(kb, args.query)
    else:
        # Run default test suite
        run_test_suite(kb)


def test_single_query(kb, query: str):
    """Test a single query with simple output"""

    print(f"\n[INFO] Testing Query: '{query}'")
    print("-" * 40)

    try:
        # Test main search
        print("\n[INFO] Search Results:")
        results = kb.search(query, top_k=10)
        display_simple_results(results)

    except Exception as e:
        print(f"[ERROR] Error during search: {e}")


def display_simple_results(results):
    """Display search results in a simple, clean format"""

    if results:
        for i, doc in enumerate(results, 1):
            content_preview = doc.page_content[:1000].replace("\n", " ")
            doc_type = doc.metadata.get("doc_type", "unknown")
            source = doc.metadata.get("source_file", "unknown")
            relevance_score = doc.metadata.get("relevance_score", "N/A")

            print(
                f"  {i}. [{doc_type}] Score: {relevance_score} - {content_preview}..."
            )
            print(f"     Source: {source}")
    else:
        print("  No results found")


def run_interactive_tests(kb):
    """Run interactive testing session"""

    print("\n[INFO] Interactive Testing Mode")
    print("Available commands:")
    print("  - Enter a query to search")
    print("  - 'stats' to view knowledge base statistics")
    print("  - 'quit' to exit")
    print("-" * 50)

    while True:
        try:
            user_input = input("\n[INPUT] Enter command: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                break

            if not user_input:
                continue

            # Handle 'stats' command
            if user_input.lower() == "stats":
                stats = kb.get_stats()
                print("\n[INFO] Knowledge Base Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue

            # Regular query
            print(f"\n[INFO] Search: '{user_input}'")
            results = kb.search(user_input, top_k=10)
            display_simple_results(results)

        except KeyboardInterrupt:
            print("\n[INFO] Exiting interactive mode...")
            break
        except Exception as e:
            print(f"[ERROR] Error: {e}")


def run_test_suite(kb):
    """Run comprehensive test suite"""

    test_cases = [
        # Error code queries (BM25 strength)
        {"query": "lỗi 606", "description": "Error code (lowercase)"},
        {"query": "LỖI 606", "description": "Error code (uppercase)"},
        {"query": "mã lỗi W02", "description": "Alphanumeric error code"},
        # Semantic queries (ChromaDB strength)
        {"query": "không nạp được tiền", "description": "Semantic: cannot topup"},
        {"query": "giao dịch bị treo", "description": "Semantic: transaction stuck"},
        # Procedure queries
        {
            "query": "hướng dẫn nạp cước trả trước",
            "description": "Procedure: prepaid topup",
        },
        {
            "query": "cách kiểm tra phí chiết khấu",
            "description": "Procedure: check discount",
        },
        # Reference queries
        {
            "query": "thẻ 30k có nhà mạng nào",
            "description": "Reference: denomination availability",
        },
        # Policy queries
        {
            "query": "quy định hủy giao dịch",
            "description": "Policy: cancellation rules",
        },
    ]

    print("\n[INFO] Running Test Suite:")
    print("=" * 50)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n#{i} {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        print("-" * 30)

        try:
            results = kb.search(test_case["query"], top_k=3)
            display_simple_results(results)
        except Exception as e:
            print(f"[ERROR] Error: {e}")


def main():
    """Main entry point with argument parsing"""

    parser = argparse.ArgumentParser(
        description="ViettelPay Knowledge Base Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_database_script.py ingest --documents-folder ./viettelpay_docs
  python build_database_script.py test --query "lỗi 606"
  python build_database_script.py test --interactive
  
Environment Variables:
  OPENAI_API_KEY: Optional API key for contextual enhancement
        """,
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest documents and build knowledge base"
    )
    ingest_parser.add_argument(
        "--documents-folder",
        default="./viettelpay_docs/raw",
        help="Directory containing Word documents (.doc/.docx files)",
    )
    ingest_parser.add_argument(
        "--chroma-dir",
        default="./knowledge_base/chroma_db",
        help="ChromaDB storage directory",
    )
    ingest_parser.add_argument(
        "--bm25-dir",
        default="./knowledge_base/bm25_index",
        help="BM25 storage directory",
    )
    ingest_parser.add_argument(
        "--embedding-model",
        default="dangvantuan/vietnamese-document-embedding",
        help="Embedding model name",
    )
    ingest_parser.add_argument(
        "--reset",
        action="store_true",
        default=True,
        help="Reset knowledge base before ingestion (default: True)",
    )
    ingest_parser.add_argument(
        "--no-reset",
        dest="reset",
        action="store_false",
        help="Do not reset existing knowledge base",
    )

    # Test command
    test_parser = subparsers.add_parser(
        "test", help="Test retrieval on existing knowledge base"
    )
    test_parser.add_argument("--query", help="Single query to test")
    test_parser.add_argument(
        "--interactive", action="store_true", help="Interactive testing mode"
    )
    test_parser.add_argument(
        "--chroma-dir",
        default="./knowledge_base/chroma_db",
        help="ChromaDB storage directory",
    )
    test_parser.add_argument(
        "--bm25-dir",
        default="./knowledge_base/bm25_index",
        help="BM25 storage directory",
    )
    test_parser.add_argument(
        "--embedding-model",
        default="dangvantuan/vietnamese-document-embedding",
        help="Embedding model name",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        success = ingest_documents(args)
        sys.exit(0 if success else 1)

    elif args.command == "test":
        test_retrieval(args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
