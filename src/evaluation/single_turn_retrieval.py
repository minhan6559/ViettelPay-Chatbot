"""
Single Turn Synthetic Retrieval Evaluation Dataset Creator for ViettelPay RAG System
Uses Google Gemini 2.0 Flash with JSON responses for better parsing
Simplified version with only MRR and hit rate evaluation (no qrels generation)
"""

import json
import os
import sys
import argparse
import time
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import re

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Add the project root to Python path so we can import from src
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import prompts (only the ones we need)
from src.evaluation.prompts import (
    QUESTION_GENERATION_PROMPT,
    QUESTION_QUALITY_CHECK_PROMPT,
    CONTEXT_QUALITY_CHECK_PROMPT,
    QUESTION_EVOLUTION_PROMPT,
)

# Import your existing knowledge base and LLM client
from src.knowledge_base.viettel_knowledge_base import ViettelKnowledgeBase
from src.llm.llm_client import LLMClientFactory, BaseLLMClient


class SingleTurnDatasetCreator:
    """Single turn synthetic evaluation dataset creator with JSON responses and all chunks processing"""

    def __init__(
        self, gemini_api_key: str, knowledge_base: ViettelKnowledgeBase = None
    ):
        """
        Initialize with Gemini API key and optional knowledge base

        Args:
            gemini_api_key: Google AI API key for Gemini
            knowledge_base: Pre-initialized ViettelKnowledgeBase instance
        """
        self.llm_client = LLMClientFactory.create_client(
            "gemini", api_key=gemini_api_key, model="gemini-2.0-flash"
        )
        self.knowledge_base = knowledge_base
        self.dataset = {
            "queries": {},
            "documents": {},
            "metadata": {
                "total_chunks_processed": 0,
                "questions_generated": 0,
                "creation_timestamp": time.time(),
            },
        }

        print("✅ SingleTurnDatasetCreator initialized with Gemini 2.0 Flash")

    def generate_json_response(
        self, prompt: str, max_retries: int = 3
    ) -> Optional[Dict]:
        """
        Generate response and parse as JSON with retries

        Args:
            prompt: Input prompt
            max_retries: Maximum number of retry attempts

        Returns:
            Parsed JSON response or None if failed
        """
        for attempt in range(max_retries):
            try:
                response = self.llm_client.generate(prompt, temperature=0.1)

                if response:
                    # Clean response text
                    response_text = response.strip()

                    # Extract JSON from response (handle cases with extra text)
                    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                    if json_match:
                        json_text = json_match.group()
                        return json.loads(json_text)
                    else:
                        # Try parsing the whole response
                        return json.loads(response_text)

            except json.JSONDecodeError as e:
                print(f"⚠️ JSON parsing error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    print(f"❌ Failed to parse JSON after {max_retries} attempts")
                    print(
                        f"Raw response: {response if 'response' in locals() else 'No response'}"
                    )

            except Exception as e:
                print(f"⚠️ API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff

        return None

    def get_all_chunks(self) -> List[Dict]:
        """
        Get ALL chunks directly from ChromaDB vectorstore (no sampling)

        Returns:
            List of all document chunks with content and metadata
        """
        print(f"📚 Retrieving ALL chunks directly from ChromaDB vectorstore...")

        if not self.knowledge_base:
            raise ValueError(
                "Knowledge base not provided. Please initialize with a ViettelKnowledgeBase instance."
            )

        try:
            # Access the ChromaDB vectorstore directly
            if (
                not hasattr(self.knowledge_base, "chroma_retriever")
                or not self.knowledge_base.chroma_retriever
            ):
                raise ValueError("ChromaDB retriever not found in knowledge base")

            # Get the vectorstore from the retriever
            vectorstore = self.knowledge_base.chroma_retriever.vectorstore

            # Get all documents directly from ChromaDB
            print("   Accessing ChromaDB collection...")
            all_docs = vectorstore.get(include=["documents", "metadatas"])

            documents = all_docs["documents"]
            metadatas = all_docs["metadatas"]

            print(f"   Found {len(documents)} documents in ChromaDB")
            print(f"   Sample document preview:")
            for i, doc in enumerate(documents[:3]):
                print(f"      Doc {i+1}: {doc[:100]}...")

            # Convert to our expected format
            all_chunks = []
            seen_content_hashes = set()

            for i, (content, metadata) in enumerate(zip(documents, metadatas)):
                # Create content hash for deduplication (just in case)
                content_hash = hash(content[:300])

                if (
                    content_hash not in seen_content_hashes
                    and len(content.strip()) > 50
                ):
                    chunk_info = {
                        "id": f"chunk_{len(all_chunks)}",
                        "content": content,
                        "metadata": metadata or {},
                        "source": "chromadb_direct",
                        "content_length": len(content),
                        "original_index": i,
                    }
                    all_chunks.append(chunk_info)
                    seen_content_hashes.add(content_hash)
                else:
                    if content_hash in seen_content_hashes:
                        print(f"   ⚠️ Skipping duplicate content at index {i}")
                    else:
                        print(
                            f"   ⚠️ Skipping short content at index {i} (length: {len(content.strip())})"
                        )

            print(f"✅ Retrieved {len(all_chunks)} unique chunks from ChromaDB")
            print(
                f"   Filtered out {len(documents) - len(all_chunks)} duplicates/short chunks"
            )

            # Sort by content length (longer chunks first, usually more informative)
            all_chunks.sort(key=lambda x: x["content_length"], reverse=True)

            # Display statistics
            avg_length = sum(chunk["content_length"] for chunk in all_chunks) / len(
                all_chunks
            )
            min_length = min(chunk["content_length"] for chunk in all_chunks)
            max_length = max(chunk["content_length"] for chunk in all_chunks)

            print(f"   📊 Chunk Statistics:")
            print(f"      • Average length: {avg_length:.0f} characters")
            print(f"      • Min length: {min_length} characters")
            print(f"      • Max length: {max_length} characters")

            return all_chunks

        except Exception as e:
            print(f"❌ Error accessing ChromaDB directly: {e}")
            print(f"   Falling back to search-based method...")
            return self._get_all_chunks_fallback()

    def _get_all_chunks_fallback(self) -> List[Dict]:
        """
        Fallback method using search queries if direct ChromaDB access fails

        Returns:
            List of document chunks retrieved via search
        """
        print(f"🔄 Using fallback search-based chunk retrieval...")

        # Use comprehensive search terms to capture most content
        comprehensive_queries = [
            "ViettelPay",
            "nạp",
            "cước",
            "giao dịch",
            "thanh toán",
            "lỗi",
            "hủy",
            "thẻ",
            "chuyển",
            "tiền",
            "quy định",
            "phí",
            "dịch vụ",
            "tài khoản",
            "ngân hàng",
            "OTP",
            "PIN",
            "mã",
            "số",
            "điện thoại",
            "internet",
            "truyền hình",
            "homephone",
            "cố định",
            "game",
            "Viettel",
            "Mobifone",
            # Add some Vietnamese words that might not be captured above
            "ứng dụng",
            "khách hàng",
            "hỗ trợ",
            "kiểm tra",
            "xác nhận",
            "bảo mật",
        ]

        all_chunks = []
        seen_content_hashes = set()

        for query in comprehensive_queries:
            try:
                # Search with large k to get as many chunks as possible
                docs = self.knowledge_base.search(query, top_k=50)

                for doc in docs:
                    # Create content hash for deduplication
                    content_hash = hash(doc.page_content[:300])

                    if (
                        content_hash not in seen_content_hashes
                        and len(doc.page_content.strip()) > 50
                    ):
                        chunk_info = {
                            "id": f"chunk_{len(all_chunks)}",
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "source": f"search_{query}",
                            "content_length": len(doc.page_content),
                        }
                        all_chunks.append(chunk_info)
                        seen_content_hashes.add(content_hash)

            except Exception as e:
                print(f"⚠️ Error searching for '{query}': {e}")
                continue

        print(f"✅ Fallback method retrieved {len(all_chunks)} unique chunks")

        # Sort by content length
        all_chunks.sort(key=lambda x: x["content_length"], reverse=True)

        return all_chunks

    def generate_questions_for_chunk(
        self, chunk: Dict, num_questions: int = 2
    ) -> List[Dict]:
        """
        Generate questions for a single chunk using Gemini with JSON response

        Args:
            chunk: Chunk dictionary with content and metadata
            num_questions: Number of questions to generate per chunk

        Returns:
            List of question dictionaries with metadata
        """
        content = chunk["content"]

        prompt = QUESTION_GENERATION_PROMPT.format(
            num_questions=num_questions, content=content
        )

        response_json = self.generate_json_response(prompt)

        if response_json and "questions" in response_json:
            questions = response_json["questions"]

            # Create question objects with metadata
            question_objects = []
            for i, question_text in enumerate(questions):
                if len(question_text.strip()) > 5:  # Filter very short questions
                    question_obj = {
                        "id": f"q_{chunk['id']}_{i}",
                        "text": question_text.strip(),
                        "source_chunk": chunk["id"],
                        "chunk_metadata": chunk["metadata"],
                        "generation_method": "gemini_json",
                    }
                    question_objects.append(question_obj)

            return question_objects
        else:
            print(f"⚠️ No valid questions generated for chunk {chunk['id']}")
            return []

    def check_context_quality(self, chunk: Dict) -> bool:
        """
        Check if a chunk is suitable for question generation

        Args:
            chunk: Chunk dictionary

        Returns:
            True if chunk should be used, False otherwise
        """
        content = chunk["content"]

        # Basic checks first
        if len(content.strip()) < 100:
            return False

        # Use Gemini for quality assessment
        prompt = CONTEXT_QUALITY_CHECK_PROMPT.format(content=content[:1000])

        response_json = self.generate_json_response(prompt)

        if response_json:
            return response_json.get("use_context", True)
        else:
            # Fallback to basic heuristics
            return len(content.strip()) > 100 and len(content.split()) > 20

    def create_complete_dataset(
        self,
        questions_per_chunk: int = 2,
        save_path: str = "evaluation_data/datasets/single_turn_retrieval/viettelpay_complete_eval_dataset.json",
        quality_check: bool = True,
    ) -> Dict:
        """
        Create complete synthetic evaluation dataset using ALL chunks

        Args:
            questions_per_chunk: Number of questions to generate per chunk
            save_path: Path to save the dataset JSON file
            quality_check: Whether to perform quality checks on chunks

        Returns:
            Complete dataset dictionary
        """
        print(f"\n🚀 Creating simplified synthetic evaluation dataset...")
        print(f"   Target: Process ALL chunks from knowledge base")
        print(f"   Questions per chunk: {questions_per_chunk}")
        print(f"   Quality check: {quality_check}")
        print(f"   Evaluation method: MRR and Hit Rates only (no qrels)")

        # Step 1: Get all chunks
        all_chunks = self.get_all_chunks()
        total_chunks = len(all_chunks)

        if total_chunks == 0:
            raise ValueError("No chunks found in knowledge base!")

        print(f"✅ Found {total_chunks} chunks to process")

        # Step 2: Quality filtering (optional)
        if quality_check:
            print(f"\n🔍 Performing quality checks on chunks...")
            quality_chunks = []

            for chunk in tqdm(all_chunks, desc="Quality checking"):
                if self.check_context_quality(chunk):
                    quality_chunks.append(chunk)
                time.sleep(0.1)  # Rate limiting

            print(
                f"✅ {len(quality_chunks)}/{total_chunks} chunks passed quality check"
            )
            chunks_to_process = quality_chunks
        else:
            chunks_to_process = all_chunks

        # Step 3: Generate questions for all chunks
        print(f"\n📝 Generating questions for {len(chunks_to_process)} chunks...")
        all_questions = []

        for chunk in tqdm(chunks_to_process, desc="Generating questions"):
            questions = self.generate_questions_for_chunk(chunk, questions_per_chunk)
            all_questions.extend(questions)
            time.sleep(0.2)  # Rate limiting for Gemini API

        print(
            f"✅ Generated {len(all_questions)} questions from {len(chunks_to_process)} chunks"
        )

        # Step 4: Populate dataset structure
        self.dataset["documents"] = {
            chunk["id"]: chunk["content"] for chunk in chunks_to_process
        }
        self.dataset["queries"] = {q["id"]: q["text"] for q in all_questions}

        # Add question metadata
        question_metadata = {
            q["id"]: {
                "source_chunk": q["source_chunk"],
                "chunk_metadata": q["chunk_metadata"],
                "generation_method": q["generation_method"],
            }
            for q in all_questions
        }

        self.dataset["question_metadata"] = question_metadata

        # Step 5: Update metadata
        self.dataset["metadata"].update(
            {
                "total_chunks_processed": len(chunks_to_process),
                "total_chunks_available": total_chunks,
                "questions_generated": len(all_questions),
                "questions_per_chunk": questions_per_chunk,
                "quality_check_enabled": quality_check,
                "evaluation_method": "mrr_hit_rates_only",
                "completion_timestamp": time.time(),
            }
        )

        # Step 6: Save dataset
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=2)

        print(f"\n✅ COMPLETE dataset created successfully!")
        print(f"   📁 Saved to: {save_path}")
        print(f"   📊 Statistics:")
        print(f"      • Chunks processed: {len(chunks_to_process)}/{total_chunks}")
        print(f"      • Questions generated: {len(all_questions)}")
        print(f"      • Evaluation method: MRR and Hit Rates only")
        print(
            f"      • Coverage: {len(chunks_to_process)/total_chunks*100:.1f}% of knowledge base"
        )

        return self.dataset

    def load_dataset(self, dataset_path: str) -> Dict:
        """Load dataset from JSON file with metadata"""
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

        metadata = self.dataset.get("metadata", {})

        print(f"📖 Loaded dataset from {dataset_path}")
        print(f"   📊 Dataset Statistics:")
        print(f"      • Queries: {len(self.dataset['queries'])}")
        print(f"      • Documents: {len(self.dataset['documents'])}")
        print(f"      • Created: {time.ctime(metadata.get('creation_timestamp', 0))}")

        return self.dataset


class SingleTurnRetrievalEvaluator:
    """Simplified retrieval evaluator with only MRR and hit rates"""

    def __init__(self, dataset: Dict, knowledge_base: ViettelKnowledgeBase):
        """
        Initialize evaluator with dataset and knowledge base

        Args:
            dataset: Evaluation dataset with queries and documents
            knowledge_base: ViettelKnowledgeBase instance to evaluate
        """
        self.dataset = dataset
        self.knowledge_base = knowledge_base
        self.results = {}

    def _match_retrieved_documents(self, retrieved_docs) -> List[str]:
        """
        Enhanced document matching with multiple strategies

        Args:
            retrieved_docs: Retrieved Document objects from knowledge base

        Returns:
            List of matched document IDs
        """
        matched_ids = []

        for doc in retrieved_docs:
            # Strategy 1: Try to find exact content match
            doc_id = self._find_exact_content_match(doc.page_content)

            if not doc_id:
                # Strategy 2: Try fuzzy content matching
                doc_id = self._find_fuzzy_content_match(doc.page_content)

            if doc_id:
                matched_ids.append(doc_id)

        return matched_ids

    def _find_exact_content_match(self, retrieved_content: str) -> Optional[str]:
        """Find exact content match"""
        for doc_id, doc_content in self.dataset["documents"].items():
            if retrieved_content.strip() == doc_content.strip():
                return doc_id
        return None

    def _find_fuzzy_content_match(
        self, retrieved_content: str, min_overlap: int = 50
    ) -> Optional[str]:
        """Find fuzzy content match with word overlap"""
        best_match_id = None
        best_overlap = 0

        retrieved_words = set(retrieved_content.lower().split())

        for doc_id, doc_content in self.dataset["documents"].items():
            doc_words = set(doc_content.lower().split())
            overlap = len(retrieved_words & doc_words)

            if overlap > best_overlap and overlap >= min_overlap:
                best_overlap = overlap
                best_match_id = doc_id

        return best_match_id

    def _safe_average(self, values: List[float]) -> float:
        """Calculate average safely handling empty lists"""
        return sum(values) / len(values) if values else 0.0

    def evaluate(self, k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        """
        Simplified evaluation with only MRR and hit rates

        This method checks if the source document (where the question was generated from)
        is retrieved among the top-k results.

        Args:
            k_values: List of k values to evaluate

        Returns:
            Dictionary with MRR and hit rate results
        """
        print(f"\n🔍 Running simplified evaluation (MRR and Hit Rates only)...")
        print(f"   📊 K values: {k_values}")
        print(f"   📚 Total queries: {len(self.dataset['queries'])}")

        # Initialize results
        hit_rates = {k: [] for k in k_values}
        rr_scores = []  # Reciprocal Rank scores for MRR calculation
        query_results = {}
        failed_queries = []

        # Process each query
        for query_id, query_text in tqdm(
            self.dataset["queries"].items(), desc="Evaluating queries"
        ):
            try:
                # Get source document from metadata - handle both single-turn and multi-turn formats
                source_chunk_id = None

                # Try question_metadata first (single-turn format)
                question_meta = self.dataset.get("question_metadata", {}).get(
                    query_id, {}
                )
                if question_meta:
                    source_chunk_id = question_meta.get("source_chunk")

                # If not found, try conversation_metadata (multi-turn format)
                if not source_chunk_id:
                    conversation_meta = self.dataset.get(
                        "conversation_metadata", {}
                    ).get(query_id, {})
                    if conversation_meta:
                        source_chunk_id = conversation_meta.get("source_chunk")

                if not source_chunk_id:
                    print(f"⚠️ No source chunk info for query {query_id}")
                    continue

                # Get retrieval results
                retrieved_docs = self.knowledge_base.search(
                    query_text, top_k=max(k_values)
                )
                retrieved_doc_ids = self._match_retrieved_documents(retrieved_docs)

                # Check if source document is in top-k for each k
                query_results[query_id] = {
                    "query": query_text,
                    "source_chunk": source_chunk_id,
                    "retrieved": retrieved_doc_ids,
                    "hit_rates": {},
                }

                # Calculate Reciprocal Rank (MRR) - once per query
                if source_chunk_id in retrieved_doc_ids:
                    source_rank = (
                        retrieved_doc_ids.index(source_chunk_id) + 1
                    )  # 1-indexed rank
                    rr_score = 1.0 / source_rank
                else:
                    rr_score = 0.0

                query_results[query_id]["rr"] = rr_score
                query_results[query_id]["source_rank"] = (
                    source_rank if rr_score > 0 else None
                )
                rr_scores.append(rr_score)

                for k in k_values:
                    top_k_docs = retrieved_doc_ids[:k]
                    hit = 1 if source_chunk_id in top_k_docs else 0
                    hit_rates[k].append(hit)
                    query_results[query_id]["hit_rates"][k] = hit

            except Exception as e:
                print(f"❌ Error evaluating query {query_id}: {e}")
                failed_queries.append((query_id, str(e)))
                continue

        # Calculate average metrics
        avg_hit_rates = {}
        avg_rr = sum(rr_scores) / len(rr_scores) if rr_scores else 0.0

        for k in k_values:
            avg_hit_rates[k] = self._safe_average(hit_rates[k])

        results = {
            "hit_rates": avg_hit_rates,
            "mrr": avg_rr,
            "per_query_results": query_results,
            "failed_queries": failed_queries,
            "summary": {
                "total_queries": len(self.dataset["queries"]),
                "evaluated_queries": len(query_results),
                "failed_queries": len(failed_queries),
                "success_rate": len(query_results) / len(self.dataset["queries"]) * 100,
                "k_values": k_values,
                "evaluation_type": "mrr_hit_rates_only",
                "evaluation_timestamp": time.time(),
            },
        }

        return results

    def print_evaluation_results(self, results: Dict):
        """Print simplified evaluation results"""
        print(f"\n📊 SIMPLIFIED EVALUATION RESULTS (MRR + Hit Rates)")
        print("=" * 60)

        print(f"\n📈 Hit Rates (Source Document Found in Top-K):")
        print(f"{'K':<5} {'Hit Rate':<12} {'Percentage':<12}")
        print("-" * 30)

        for k in sorted(results["hit_rates"].keys()):
            hit_rate = results["hit_rates"][k]
            percentage = hit_rate * 100
            print(f"{k:<5} {hit_rate:<12.4f} {percentage:<12.1f}%")

        # Display MRR separately since it's not k-dependent
        mrr = results["mrr"]
        print(f"\n📊 Mean Reciprocal Rank (MRR): {mrr:.4f}")
        print(f"   • MRR measures the average reciprocal rank of the source document")
        print(f"   • Higher is better (max = 1.0 if all sources are rank 1)")

        print(f"\n📊 Hit Rate Summary:")
        for k in sorted(results["hit_rates"].keys()):
            hit_rate = results["hit_rates"][k]
            percentage = hit_rate * 100
            print(
                f"   • Top-{k}: {percentage:.1f}% of questions find their source document"
            )

        # Summary stats
        summary = results["summary"]
        print(f"\n📋 Evaluation Summary:")
        print(f"   • Total queries: {summary['total_queries']}")
        print(f"   • Successfully evaluated: {summary['evaluated_queries']}")
        print(f"   • Failed queries: {summary['failed_queries']}")
        print(f"   • Success rate: {summary['success_rate']:.1f}%")
        print(f"   • Evaluation type: {summary['evaluation_type']}")

        # Simple interpretation
        avg_hit_rate_5 = results["hit_rates"].get(5, 0)
        mrr = results["mrr"]
        print(f"\n🎯 Quick Interpretation:")
        if avg_hit_rate_5 > 0.8:
            print(
                f"   ✅ Excellent: {avg_hit_rate_5*100:.1f}% hit rate@5, MRR = {mrr:.3f}"
            )
        elif avg_hit_rate_5 > 0.6:
            print(f"   👍 Good: {avg_hit_rate_5*100:.1f}% hit rate@5, MRR = {mrr:.3f}")
        elif avg_hit_rate_5 > 0.4:
            print(f"   ⚠️ Fair: {avg_hit_rate_5*100:.1f}% hit rate@5, MRR = {mrr:.3f}")
        else:
            print(f"   ❌ Poor: {avg_hit_rate_5*100:.1f}% hit rate@5, MRR = {mrr:.3f}")


def main():
    """Main function with argument parsing for separate operations"""
    parser = argparse.ArgumentParser(
        description="ViettelPay Retrieval Evaluation Dataset Creator (Simplified)"
    )
    parser.add_argument(
        "--mode",
        choices=["create", "evaluate", "both"],
        default="both",
        help="Mode: create dataset, evaluate only, or both",
    )
    parser.add_argument(
        "--dataset-path",
        default="evaluation_data/datasets/single_turn_retrieval/viettelpay_complete_eval.json",
        help="Path to dataset file",
    )
    parser.add_argument(
        "--results-path",
        default="evaluation_data/results/single_turn_retrieval/viettelpay_eval_results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--questions-per-chunk",
        type=int,
        default=3,
        help="Number of questions per chunk",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        help="K values for evaluation",
    )
    parser.add_argument(
        "--quality-check",
        action="store_true",
        help="Enable quality checking for chunks",
    )
    parser.add_argument(
        "--knowledge-base-path",
        default="./knowledge_base",
        help="Path to knowledge base",
    )

    args = parser.parse_args()

    # Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        print("❌ Please set GEMINI_API_KEY environment variable")
        return

    try:
        # Initialize knowledge base
        print("🔧 Initializing ViettelPay knowledge base...")
        kb = ViettelKnowledgeBase()
        if not kb.load_knowledge_base(args.knowledge_base_path):
            print(
                "❌ Failed to load knowledge base. Please run build_database_script.py first."
            )
            return

        # Create dataset if requested
        if args.mode in ["create", "both"]:
            print(f"\n🎯 Creating synthetic evaluation dataset...")
            creator = SingleTurnDatasetCreator(GEMINI_API_KEY, kb)

            dataset = creator.create_complete_dataset(
                questions_per_chunk=args.questions_per_chunk,
                save_path=args.dataset_path,
                quality_check=args.quality_check,
            )

        # Evaluate if requested
        if args.mode in ["evaluate", "both"]:
            print(f"\n⚡ Evaluating retrieval performance...")

            # Load dataset if not created in this run
            if args.mode == "evaluate":
                if not os.path.exists(args.dataset_path):
                    print(f"❌ Dataset file not found: {args.dataset_path}")
                    return

                creator = SingleTurnDatasetCreator(GEMINI_API_KEY, kb)
                dataset = creator.load_dataset(args.dataset_path)

            # Run evaluation
            evaluator = SingleTurnRetrievalEvaluator(dataset, kb)
            results = evaluator.evaluate(k_values=args.k_values)
            evaluator.print_evaluation_results(results)

            # Save results
            if args.results_path:
                with open(args.results_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"\n💾 Results saved to: {args.results_path}")

        print(f"\n✅ Operation completed successfully!")
        print(f"\n💡 Next steps:")
        print(f"   1. Review the MRR and hit rate results")
        print(f"   2. Identify queries with low performance")
        print(f"   3. Optimize your retrieval system")
        print(f"   4. Re-run evaluation to measure progress")

    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
