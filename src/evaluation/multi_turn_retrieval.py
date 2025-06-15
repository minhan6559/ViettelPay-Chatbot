"""
Multi-Turn Conversation Retrieval Evaluation for ViettelPay RAG System
Generates multi-turn conversations and evaluates retrieval performance
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

# Import existing components
from src.evaluation.prompts import MULTI_TURN_CONVERSATION_GENERATION_PROMPT
from src.knowledge_base.viettel_knowledge_base import ViettelKnowledgeBase
from src.evaluation.single_turn_retrieval import SingleTurnRetrievalEvaluator
from src.llm.llm_client import LLMClientFactory, BaseLLMClient
from src.agent.nodes import query_enhancement_node, ViettelPayState
from langchain_core.messages import HumanMessage


class MultiTurnDatasetCreator:
    """Multi-turn conversation dataset creator for ViettelPay evaluation"""

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
            "conversations": {},
            "documents": {},
            "metadata": {
                "total_chunks_processed": 0,
                "conversations_generated": 0,
                "creation_timestamp": time.time(),
            },
        }

        print("✅ MultiTurnDatasetCreator initialized with Gemini 2.0 Flash")

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
        Get ALL chunks directly from ChromaDB vectorstore
        Reuse the same method from single-turn evaluation

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

            # Convert to our expected format
            all_chunks = []
            seen_content_hashes = set()

            for i, (content, metadata) in enumerate(zip(documents, metadatas)):
                # Create content hash for deduplication
                content_hash = hash(content[:300])

                if (
                    content_hash not in seen_content_hashes
                    and len(content.strip()) > 100
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

            print(f"✅ Retrieved {len(all_chunks)} unique chunks from ChromaDB")

            # Sort by content length (longer chunks first)
            all_chunks.sort(key=lambda x: x["content_length"], reverse=True)

            return all_chunks

        except Exception as e:
            print(f"❌ Error accessing ChromaDB directly: {e}")
            return []

    def generate_conversations_for_chunk(
        self, chunk: Dict, num_conversations: int = 2
    ) -> List[Dict]:
        """
        Generate multi-turn conversations for a single chunk using Gemini

        Args:
            chunk: Chunk dictionary with content and metadata
            num_conversations: Number of conversations to generate per chunk

        Returns:
            List of conversation dictionaries
        """
        content = chunk["content"]

        prompt = MULTI_TURN_CONVERSATION_GENERATION_PROMPT.format(
            num_conversations=num_conversations, content=content
        )

        response_json = self.generate_json_response(prompt)

        if response_json and "conversations" in response_json:
            conversations = response_json["conversations"]

            # Create conversation objects with metadata
            conversation_objects = []
            for i, conversation in enumerate(conversations):
                if len(conversation.get("turns", [])) >= 2:  # At least 2 turns
                    conversation_obj = {
                        "id": f"conv_{chunk['id']}_{i}",
                        "turns": conversation["turns"],
                        "conversation_type": conversation.get("type", "general"),
                        "source_chunk": chunk["id"],
                        "chunk_metadata": chunk["metadata"],
                        "generation_method": "gemini_json",
                    }
                    conversation_objects.append(conversation_obj)

            return conversation_objects
        else:
            print(f"⚠️ No valid conversations generated for chunk {chunk['id']}")
            return []

    def create_multi_turn_dataset(
        self,
        conversations_per_chunk: int = 2,
        save_path: str = "evaluation_data/datasets/multi_turn_retrieval/viettelpay_multiturn_conversations.json",
    ) -> Dict:
        """
        Create multi-turn conversation dataset using ALL chunks

        Args:
            conversations_per_chunk: Number of conversations to generate per chunk
            save_path: Path to save the dataset JSON file

        Returns:
            Complete dataset dictionary with conversations
        """
        print(f"\n🚀 Creating multi-turn conversation dataset...")
        print(f"   Target: Process ALL chunks from knowledge base")
        print(f"   Conversations per chunk: {conversations_per_chunk}")

        # Step 1: Get all chunks
        all_chunks = self.get_all_chunks()
        total_chunks = len(all_chunks)

        if total_chunks == 0:
            raise ValueError("No chunks found in knowledge base!")

        print(f"✅ Found {total_chunks} chunks to process")

        # Step 2: Generate conversations for all chunks
        print(f"\n💬 Generating conversations for {total_chunks} chunks...")
        all_conversations = []

        for chunk in tqdm(all_chunks, desc="Generating conversations"):
            conversations = self.generate_conversations_for_chunk(
                chunk, conversations_per_chunk
            )
            all_conversations.extend(conversations)
            time.sleep(0.2)  # Rate limiting for Gemini API

        # Step 3: Populate dataset structure
        self.dataset["documents"] = {
            chunk["id"]: chunk["content"] for chunk in all_chunks
        }
        self.dataset["conversations"] = {
            conv["id"]: {
                "turns": conv["turns"],
                "conversation_type": conv["conversation_type"],
                "source_chunk": conv["source_chunk"],
                "chunk_metadata": conv["chunk_metadata"],
                "generation_method": conv["generation_method"],
            }
            for conv in all_conversations
        }

        # Step 4: Update metadata
        self.dataset["metadata"].update(
            {
                "total_chunks_processed": total_chunks,
                "conversations_generated": len(all_conversations),
                "conversations_per_chunk": conversations_per_chunk,
                "completion_timestamp": time.time(),
            }
        )

        # Step 5: Save dataset
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Multi-turn conversation dataset created successfully!")
        print(f"   📁 Saved to: {save_path}")
        print(f"   📊 Statistics:")
        print(f"      • Chunks processed: {total_chunks}")
        print(f"      • Conversations generated: {len(all_conversations)}")
        print(
            f"      • Avg conversations per chunk: {len(all_conversations)/total_chunks:.1f}"
        )

        return self.dataset


class ConversationEnhancer:
    """Convert multi-turn conversations to enhanced queries using existing query enhancement"""

    def __init__(self, gemini_api_key: str):
        """Initialize with Gemini API key for query enhancement"""
        self.llm_client = LLMClientFactory.create_client(
            "gemini", api_key=gemini_api_key, model="gemini-2.0-flash-lite"
        )
        print("✅ ConversationEnhancer initialized")

    def enhance_conversation(self, conversation_turns: List[Dict]) -> str:
        """
        Convert a multi-turn conversation to an enhanced query

        Args:
            conversation_turns: List of turn dictionaries with role and content

        Returns:
            Enhanced query string
        """
        try:
            # Create messages in the format expected by query_enhancement_node
            messages = []
            for turn in conversation_turns:
                if turn["role"] == "user":
                    messages.append(HumanMessage(content=turn["content"]))

            # Create a mock state for the query enhancement node
            state = ViettelPayState(messages=messages)

            # Use the existing query enhancement node
            enhanced_state = query_enhancement_node(state, self.llm_client)

            enhanced_query = enhanced_state.get("enhanced_query", "")

            if not enhanced_query:
                # Fallback: concatenate all user messages
                user_messages = [
                    turn["content"]
                    for turn in conversation_turns
                    if turn["role"] == "user"
                ]
                enhanced_query = " ".join(user_messages)

            return enhanced_query

        except Exception as e:
            print(f"❌ Error enhancing conversation: {e}")
            # Fallback: concatenate all user messages
            user_messages = [
                turn["content"] for turn in conversation_turns if turn["role"] == "user"
            ]
            return " ".join(user_messages)

    def convert_dataset_to_single_turn_format(
        self,
        multi_turn_dataset: Dict,
        save_path: str = "evaluation_data/datasets/multi_turn_retrieval/viettelpay_multiturn_enhanced.json",
    ) -> Dict:
        """
        Convert multi-turn conversation dataset to single-turn format with enhanced queries

        Args:
            multi_turn_dataset: Multi-turn conversation dataset
            save_path: Path to save the converted dataset

        Returns:
            Single-turn format dataset
        """
        print(f"\n🔄 Converting multi-turn conversations to enhanced queries...")

        conversations = multi_turn_dataset["conversations"]
        documents = multi_turn_dataset["documents"]

        # Initialize single-turn format dataset
        single_turn_dataset = {
            "queries": {},
            "documents": documents,
            "conversation_metadata": {},
            "metadata": {
                "total_conversations_processed": len(conversations),
                "enhanced_queries_generated": 0,
                "conversion_timestamp": time.time(),
                "original_dataset_metadata": multi_turn_dataset.get("metadata", {}),
            },
        }

        enhanced_count = 0

        # Process each conversation
        for conv_id, conv_data in tqdm(
            conversations.items(), desc="Enhancing conversations"
        ):
            try:
                # Extract turns
                turns = conv_data["turns"]

                # Enhance conversation to single query
                enhanced_query = self.enhance_conversation(turns)

                if enhanced_query and len(enhanced_query.strip()) > 5:
                    single_turn_dataset["queries"][conv_id] = enhanced_query
                    single_turn_dataset["conversation_metadata"][conv_id] = {
                        "original_conversation": turns,
                        "conversation_type": conv_data.get(
                            "conversation_type", "general"
                        ),
                        "source_chunk": conv_data["source_chunk"],
                        "chunk_metadata": conv_data.get("chunk_metadata", {}),
                        "generation_method": conv_data.get(
                            "generation_method", "unknown"
                        ),
                    }
                    enhanced_count += 1

                time.sleep(0.1)  # Small delay for rate limiting

            except Exception as e:
                print(f"⚠️ Error processing conversation {conv_id}: {e}")
                continue

        # Update metadata
        single_turn_dataset["metadata"]["enhanced_queries_generated"] = enhanced_count

        # Save converted dataset
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(single_turn_dataset, f, ensure_ascii=False, indent=2)

        print(f"✅ Conversion completed successfully!")
        print(f"   📁 Saved to: {save_path}")
        print(f"   📊 Statistics:")
        print(f"      • Conversations processed: {len(conversations)}")
        print(f"      • Enhanced queries generated: {enhanced_count}")
        print(f"      • Success rate: {enhanced_count/len(conversations)*100:.1f}%")

        return single_turn_dataset


class MultiTurnEvaluator:
    """Extended evaluator for multi-turn conversation retrieval with additional analysis"""

    def __init__(self, dataset: Dict, knowledge_base: ViettelKnowledgeBase):
        """
        Initialize evaluator with dataset and knowledge base

        Args:
            dataset: Evaluation dataset in single-turn format (from converted multi-turn)
            knowledge_base: ViettelKnowledgeBase instance to evaluate
        """
        self.dataset = dataset
        self.knowledge_base = knowledge_base
        self.single_turn_evaluator = SingleTurnRetrievalEvaluator(
            dataset, knowledge_base
        )

    def _get_conversation_metadata(self, query_id: str) -> Dict:
        """
        Get conversation metadata for a query, handling both formats

        Args:
            query_id: Query identifier

        Returns:
            Metadata dictionary
        """
        # First try conversation_metadata (multi-turn format)
        conversation_metadata = self.dataset.get("conversation_metadata", {})
        if query_id in conversation_metadata:
            return conversation_metadata[query_id]

        # Fallback to question_metadata (single-turn format)
        question_metadata = self.dataset.get("question_metadata", {})
        if query_id in question_metadata:
            # Convert single-turn format to multi-turn format for consistency
            meta = question_metadata[query_id]
            return {
                "conversation_type": "single_turn",
                "source_chunk": meta.get("source_chunk"),
                "original_conversation": [
                    {"role": "user", "content": self.dataset["queries"][query_id]}
                ],
                "chunk_metadata": meta.get("chunk_metadata", {}),
                "generation_method": meta.get("generation_method", "unknown"),
            }

        return {}

    def evaluate_multi_turn_performance(
        self, k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict:
        """
        Evaluate multi-turn conversation retrieval performance

        Args:
            k_values: List of k values to evaluate

        Returns:
            Dictionary with evaluation results and multi-turn specific analysis
        """
        print(f"\n🔍 Running multi-turn conversation evaluation...")

        # Step 1: Run standard single-turn evaluation
        base_results = self.single_turn_evaluator.evaluate(k_values)

        # Step 2: Add multi-turn specific analysis
        # Analyze by conversation type
        results_by_type = defaultdict(
            lambda: {"hit_rates": {k: [] for k in k_values}, "rr_scores": []}
        )

        for query_id, query_result in base_results["per_query_results"].items():
            conv_meta = self._get_conversation_metadata(query_id)
            conv_type = conv_meta.get("conversation_type", "unknown")

            # Add to type-specific results
            results_by_type[conv_type]["rr_scores"].append(query_result.get("rr", 0))
            for k in k_values:
                hit_rate = query_result.get("hit_rates", {}).get(k, 0)
                results_by_type[conv_type]["hit_rates"][k].append(hit_rate)

        # Calculate averages by conversation type
        type_analysis = {}
        for conv_type, type_results in results_by_type.items():
            type_analysis[conv_type] = {
                "hit_rates": {
                    k: sum(hits) / len(hits) if hits else 0
                    for k, hits in type_results["hit_rates"].items()
                },
                "mrr": (
                    sum(type_results["rr_scores"]) / len(type_results["rr_scores"])
                    if type_results["rr_scores"]
                    else 0
                ),
                "total_conversations": len(type_results["rr_scores"]),
            }

        # Analyze conversation length impact
        turn_length_analysis = self._analyze_by_conversation_length(
            base_results, k_values
        )

        # Combine results
        multi_turn_results = {
            **base_results,  # Include all base results
            "conversation_type_analysis": type_analysis,
            "turn_length_analysis": turn_length_analysis,
            "multi_turn_metadata": {
                "evaluation_type": "multi_turn_conversation",
                "conversation_types": list(type_analysis.keys()),
                "total_conversation_types": len(type_analysis),
            },
        }

        return multi_turn_results

    def _analyze_by_conversation_length(
        self, base_results: Dict, k_values: List[int]
    ) -> Dict:
        """Analyze performance by conversation turn length"""

        length_analysis = defaultdict(
            lambda: {"hit_rates": {k: [] for k in k_values}, "rr_scores": []}
        )

        for query_id, query_result in base_results["per_query_results"].items():
            conv_meta = self._get_conversation_metadata(query_id)
            original_conv = conv_meta.get("original_conversation", [])
            turn_count = len(
                [turn for turn in original_conv if turn.get("role") == "user"]
            )

            # Categorize by turn length
            if turn_count == 1:
                length_category = "1_turn"  # Single-turn questions
            elif turn_count == 2:
                length_category = "2_turns"
            elif turn_count == 3:
                length_category = "3_turns"
            elif turn_count >= 4:
                length_category = "4+_turns"
            else:
                length_category = "unknown_turns"

            # Add to length-specific results
            length_analysis[length_category]["rr_scores"].append(
                query_result.get("rr", 0)
            )
            for k in k_values:
                hit_rate = query_result.get("hit_rates", {}).get(k, 0)
                length_analysis[length_category]["hit_rates"][k].append(hit_rate)

        # Calculate averages by turn length
        final_length_analysis = {}
        for length_cat, length_results in length_analysis.items():
            final_length_analysis[length_cat] = {
                "hit_rates": {
                    k: sum(hits) / len(hits) if hits else 0
                    for k, hits in length_results["hit_rates"].items()
                },
                "mrr": (
                    sum(length_results["rr_scores"]) / len(length_results["rr_scores"])
                    if length_results["rr_scores"]
                    else 0
                ),
                "total_conversations": len(length_results["rr_scores"]),
            }

        return final_length_analysis

    def print_multi_turn_results(self, results: Dict):
        """Print multi-turn evaluation results with additional analysis"""

        # Print base results first
        self.single_turn_evaluator.print_evaluation_results(results)

        # Print multi-turn specific analysis
        print(f"\n🔍 MULTI-TURN SPECIFIC ANALYSIS")
        print("=" * 60)

        # Conversation type analysis
        type_analysis = results.get("conversation_type_analysis", {})
        if type_analysis:
            print(f"\n📊 Performance by Conversation Type:")
            print(f"{'Type':<20} {'MRR':<8} {'Hit@5':<8} {'Count':<8}")
            print("-" * 50)

            for conv_type, analysis in type_analysis.items():
                mrr = analysis["mrr"]
                hit_at_5 = analysis["hit_rates"].get(5, 0) * 100
                count = analysis["total_conversations"]
                print(f"{conv_type:<20} {mrr:<8.3f} {hit_at_5:<8.1f}% {count:<8}")

        # Turn length analysis
        length_analysis = results.get("turn_length_analysis", {})
        if length_analysis:
            print(f"\n📊 Performance by Conversation Length:")
            print(f"{'Length':<12} {'MRR':<8} {'Hit@5':<8} {'Count':<8}")
            print("-" * 40)

            for length_cat, analysis in length_analysis.items():
                mrr = analysis["mrr"]
                hit_at_5 = analysis["hit_rates"].get(5, 0) * 100
                count = analysis["total_conversations"]
                print(f"{length_cat:<12} {mrr:<8.3f} {hit_at_5:<8.1f}% {count:<8}")

        print(f"\n💡 Multi-Turn Insights:")

        # Best performing conversation type
        if type_analysis:
            best_type = max(type_analysis.keys(), key=lambda k: type_analysis[k]["mrr"])
            worst_type = min(
                type_analysis.keys(), key=lambda k: type_analysis[k]["mrr"]
            )
            print(
                f"   • Best conversation type: {best_type} (MRR: {type_analysis[best_type]['mrr']:.3f})"
            )
            print(
                f"   • Worst conversation type: {worst_type} (MRR: {type_analysis[worst_type]['mrr']:.3f})"
            )

        # Turn length insights
        if length_analysis:
            best_length = max(
                length_analysis.keys(), key=lambda k: length_analysis[k]["mrr"]
            )
            print(
                f"   • Best performing length: {best_length} (MRR: {length_analysis[best_length]['mrr']:.3f})"
            )


def main():
    """Main function for multi-turn conversation evaluation"""
    parser = argparse.ArgumentParser(
        description="ViettelPay Multi-Turn Conversation Retrieval Evaluation"
    )
    parser.add_argument(
        "--mode",
        choices=["create", "enhance", "evaluate", "full"],
        default="full",
        help="Mode: create conversations, enhance to queries, evaluate, or full pipeline",
    )
    parser.add_argument(
        "--conversations-dataset",
        default="evaluation_data/datasets/multi_turn_retrieval/viettelpay_multiturn_conversations.json",
        help="Path to multi-turn conversations dataset",
    )
    parser.add_argument(
        "--enhanced-dataset",
        default="evaluation_data/datasets/multi_turn_retrieval/viettelpay_multiturn_enhanced.json",
        help="Path to enhanced queries dataset",
    )
    parser.add_argument(
        "--results-path",
        default="evaluation_data/results/multi_turn_retrieval/viettelpay_multiturn_results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--conversations-per-chunk",
        type=int,
        default=3,
        help="Number of conversations per chunk",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        help="K values for evaluation",
    )
    parser.add_argument(
        "--knowledge-base-path",
        default="./knowledge_base",
        help="Path to knowledge base",
    )

    args = parser.parse_args()

    # Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    if not GOOGLE_API_KEY:
        print("❌ Please set GOOGLE_API_KEY environment variable")
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

        # Step 1: Create multi-turn conversations if requested
        if args.mode in ["create", "full"]:
            print(f"\n🎯 Creating multi-turn conversation dataset...")
            creator = MultiTurnDatasetCreator(GOOGLE_API_KEY, kb)

            conversations_dataset = creator.create_multi_turn_dataset(
                conversations_per_chunk=args.conversations_per_chunk,
                save_path=args.conversations_dataset,
            )

        # Step 2: Enhance conversations to queries if requested
        if args.mode in ["enhance", "full"]:
            print(f"\n⚡ Converting conversations to enhanced queries...")

            # Load conversations if not created in this run
            if args.mode == "enhance":
                if not os.path.exists(args.conversations_dataset):
                    print(
                        f"❌ Conversations dataset not found: {args.conversations_dataset}"
                    )
                    return

                with open(args.conversations_dataset, "r", encoding="utf-8") as f:
                    conversations_dataset = json.load(f)

            # Enhance conversations
            enhancer = ConversationEnhancer(GOOGLE_API_KEY)
            enhanced_dataset = enhancer.convert_dataset_to_single_turn_format(
                conversations_dataset, args.enhanced_dataset
            )

        # Step 3: Evaluate if requested
        if args.mode in ["evaluate", "full"]:
            print(f"\n📊 Evaluating multi-turn conversation retrieval...")

            # Load enhanced dataset if not created in this run
            if args.mode == "evaluate":
                if not os.path.exists(args.enhanced_dataset):
                    print(f"❌ Enhanced dataset not found: {args.enhanced_dataset}")
                    return

                with open(args.enhanced_dataset, "r", encoding="utf-8") as f:
                    enhanced_dataset = json.load(f)

            # Run evaluation
            evaluator = MultiTurnEvaluator(enhanced_dataset, kb)
            results = evaluator.evaluate_multi_turn_performance(k_values=args.k_values)
            evaluator.print_multi_turn_results(results)

            # Save results
            if args.results_path:
                with open(args.results_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"\n💾 Results saved to: {args.results_path}")

        print(f"\n✅ Multi-turn evaluation completed successfully!")
        print(f"\n💡 Next steps:")
        print(f"   1. Compare multi-turn vs single-turn performance")
        print(f"   2. Analyze conversation types that work best")
        print(f"   3. Optimize query enhancement for multi-turn scenarios")

    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
