"""
ViettelPay Knowledge Base Management Script

This script uses LangChain's built-in components:
- Same content for both ChromaDB and BM25
- EnsembleRetriever for fusion
- Pickle persistence for BM25
- Simple output format

Usage:
    python build_database_script.py ingest --data-dir ./viettelpay_docs/processed
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


def setup_file_paths(data_dir: str = "./viettelpay_docs/processed") -> dict:
    """Setup file paths for knowledge base construction"""

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

    return file_paths


def validate_data_files(file_paths: dict) -> bool:
    """Validate that required data files exist"""

    missing_files = []
    for file_type, file_path in file_paths.items():
        if not os.path.exists(file_path):
            missing_files.append(f"{file_type}: {file_path}")

    if missing_files:
        print("[ERROR] Missing required data files:")
        for missing in missing_files:
            print(f"   - {missing}")
        return False

    print("[SUCCESS] All required data files found")
    return True


def ingest_documents(args):
    """Ingest documents and build knowledge base"""

    print("=" * 60)
    print("[INFO] INGESTING DOCUMENTS AND BUILDING KNOWLEDGE BASE")
    print("=" * 60)

    # Setup file paths
    file_paths = setup_file_paths(args.data_dir)

    # Validate files exist
    if not validate_data_files(file_paths):
        sys.exit(1)

    # Build knowledge base
    kb = ViettelKnowledgeBase(embedding_model=args.embedding_model)

    try:
        # Create persist directory from chroma_dir
        persist_dir = os.path.dirname(args.chroma_dir) or "./knowledge_base"

        # Build knowledge base
        ensemble_retriever = kb.build_knowledge_base(
            file_paths=file_paths,
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
    ensemble_retriever = kb.load_knowledge_base(persist_dir=persist_dir)

    if not ensemble_retriever:
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
        print("\n[INFO] Main Search Results:")
        results = kb.search(query, k=5)
        display_simple_results(results)

        # Test specialized searches if applicable
        if any(err in query.lower() for err in ["lỗi", "error", "mã"]):
            print("\n[INFO] Error Code Search:")
            error_results = kb.search_by_error_code(query.split()[-1], k=3)
            display_simple_results(error_results)

        if any(proc in query.lower() for proc in ["hướng dẫn", "guide", "bước"]):
            print("\n[INFO] Procedure Search:")
            proc_results = kb.search_procedures(query, k=3)
            display_simple_results(proc_results)

    except Exception as e:
        print(f"[ERROR] Error during search: {e}")


def display_simple_results(results):
    """Display search results in a simple, clean format"""

    if results:
        for i, doc in enumerate(results, 1):
            content_preview = doc.page_content[:150].replace("\n", " ")
            doc_type = doc.metadata.get("doc_type", "unknown")
            source = doc.metadata.get("source_file", "unknown")

            print(f"  {i}. [{doc_type}] {content_preview}...")
            print(f"     Source: {source}")
    else:
        print("  No results found")


def run_interactive_tests(kb):
    """Run interactive testing session"""

    print("\n[INFO] Interactive Testing Mode")
    print("Available commands:")
    print("  - Enter a query to search")
    print("  - 'error <code>' for error code search")
    print("  - 'procedure <query>' for procedure search")
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

            # Handle 'error <code>' command
            if user_input.lower().startswith("error "):
                error_code = user_input.split()[1]
                print(f"\n[INFO] Error Code Search: '{error_code}'")
                results = kb.search_by_error_code(error_code, k=5)
                display_simple_results(results)
                continue

            # Handle 'procedure <query>' command
            if user_input.lower().startswith("procedure "):
                proc_query = user_input[10:].strip()
                print(f"\n[INFO] Procedure Search: '{proc_query}'")
                results = kb.search_procedures(proc_query, k=5)
                display_simple_results(results)
                continue

            # Regular query
            print(f"\n[INFO] Search: '{user_input}'")
            results = kb.search(user_input, k=5)
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
            results = kb.search(test_case["query"], k=3)
            display_simple_results(results)
        except Exception as e:
            print(f"[ERROR] Error: {e}")

    # Test specialized methods
    print("\n[INFO] Testing Specialized Methods:")
    print("-" * 30)

    try:
        error_results = kb.search_by_error_code("606", k=2)
        print(f"[SUCCESS] Error code search: {len(error_results)} results")

        procedure_results = kb.search_procedures("nạp tiền", k=2)
        print(f"[SUCCESS] Procedure search: {len(procedure_results)} results")

        definition_results = kb.search_definitions("người lập giao dịch", k=2)
        print(f"[SUCCESS] Definition search: {len(definition_results)} results")

    except Exception as e:
        print(f"[ERROR] Error testing specialized methods: {e}")


def main():
    """Main entry point with argument parsing"""

    parser = argparse.ArgumentParser(
        description="ViettelPay Knowledge Base Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_database_script.py ingest --data-dir ./viettelpay_docs/processed
  python build_database_script.py test --query "lỗi 606"
  python build_database_script.py test --interactive
        """,
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest documents and build knowledge base"
    )
    ingest_parser.add_argument(
        "--data-dir",
        default="./viettelpay_docs/processed",
        help="Directory containing data files",
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
