"""
Simplified Intent Classification Evaluation for ViettelPay AI Agent
Removed pattern-based generation, improved chunk mixing, and configurable conversations per chunk
"""

import json
import os
import sys
import argparse
import time
import random
from typing import Dict, List, Optional
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
from tqdm import tqdm
import re
import numpy as np

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Add the project root to Python path so we can import from src
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import existing components
from src.evaluation.prompts import INTENT_CLASSIFICATION_CONVERSATION_GENERATION_PROMPT
from src.knowledge_base.viettel_knowledge_base import ViettelKnowledgeBase
from src.llm.llm_client import LLMClientFactory
from src.agent.nodes import classify_intent_node, ViettelPayState
from langchain_core.messages import HumanMessage


class IntentDatasetCreator:
    """Simplified intent classification dataset creator with two strategies"""

    def __init__(
        self, gemini_api_key: str, knowledge_base: ViettelKnowledgeBase = None
    ):
        """Initialize with Gemini API key and optional knowledge base"""
        self.llm_client = LLMClientFactory.create_client(
            "gemini", api_key=gemini_api_key, model="gemini-2.0-flash"
        )
        self.knowledge_base = knowledge_base
        self.dataset = {
            "conversations": {},
            "generation_methods": {},
            "intent_distribution": {},
            "metadata": {
                "total_conversations": 0,
                "total_user_messages": 0,
                "creation_timestamp": time.time(),
            },
        }

        print("✅ IntentDatasetCreator initialized (simplified version)")

    def generate_json_response(
        self, prompt: str, max_retries: int = 3
    ) -> Optional[Dict]:
        """Generate response and parse as JSON with retries"""
        for attempt in range(max_retries):
            try:
                response = self.llm_client.generate(prompt, temperature=0.1)

                if response:
                    response_text = response.strip()
                    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                    if json_match:
                        json_text = json_match.group()
                        return json.loads(json_text)
                    else:
                        return json.loads(response_text)

            except json.JSONDecodeError as e:
                print(f"⚠️ JSON parsing error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    print(f"❌ Failed to parse JSON after {max_retries} attempts")

            except Exception as e:
                print(f"⚠️ API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)

        return None

    def get_all_chunks(self) -> List[Dict]:
        """Get ALL chunks from ChromaDB vectorstore"""
        print(f"📚 Retrieving ALL chunks from ChromaDB vectorstore...")

        if not self.knowledge_base:
            raise ValueError("Knowledge base not provided.")

        try:
            if (
                not hasattr(self.knowledge_base, "chroma_retriever")
                or not self.knowledge_base.chroma_retriever
            ):
                raise ValueError("ChromaDB retriever not found in knowledge base")

            vectorstore = self.knowledge_base.chroma_retriever.vectorstore
            all_docs = vectorstore.get(include=["documents", "metadatas"])

            documents = all_docs["documents"]
            metadatas = all_docs["metadatas"]

            all_chunks = []
            seen_content_hashes = set()

            for i, (content, metadata) in enumerate(zip(documents, metadatas)):
                content_hash = hash(content[:300])

                if (
                    content_hash not in seen_content_hashes
                    and len(content.strip()) > 100
                ):
                    chunk_info = {
                        "id": f"chunk_{len(all_chunks)}",
                        "content": content,
                        "metadata": metadata or {},
                    }
                    all_chunks.append(chunk_info)
                    seen_content_hashes.add(content_hash)

            print(f"✅ Retrieved {len(all_chunks)} unique chunks from ChromaDB")
            return all_chunks

        except Exception as e:
            print(f"❌ Error accessing ChromaDB: {e}")
            return []

    def generate_single_chunk_conversations(
        self, chunk: Dict, num_conversations: int = 3
    ) -> List[Dict]:
        """Generate conversations from single chunk"""
        content = chunk["content"]

        generation_instruction = "Tạo cuộc hội thoại tập trung vào chủ đề chính của tài liệu. Bao gồm cả các intent phổ biến như greeting, unclear, human_request để tăng tính đa dạng"

        prompt = INTENT_CLASSIFICATION_CONVERSATION_GENERATION_PROMPT.format(
            num_conversations=num_conversations,
            content=content,
            generation_instruction=generation_instruction,
        )

        response_json = self.generate_json_response(prompt)

        if response_json and "conversations" in response_json:
            conversations = response_json["conversations"]
            valid_conversations = []

            for i, conversation in enumerate(conversations):
                if "turns" in conversation and len(conversation["turns"]) >= 1:
                    valid_turns = []
                    for turn in conversation["turns"]:
                        if "user" in turn and "intent" in turn:
                            valid_turns.append(turn)

                    if valid_turns:
                        conv_obj = {
                            "id": f"single_{chunk['id']}_{i}",
                            "turns": valid_turns,
                            "generation_method": "single_chunk",
                            "source_chunks": [chunk["id"]],
                            "chunk_metadata": [chunk["metadata"]],
                        }
                        valid_conversations.append(conv_obj)
            return valid_conversations
        else:
            print(f"⚠️ No valid conversations generated for chunk {chunk['id']}")
            return []

    def generate_multi_chunk_conversations(
        self, chunks: List[Dict], num_conversations: int = 3
    ) -> List[Dict]:
        """Generate conversations from multiple chunks (2-3 chunks)"""
        # Combine content from multiple chunks
        combined_content = ""
        for i, chunk in enumerate(chunks):
            combined_content += f"\n\n--- Chủ đề {i+1} ---\n" + chunk["content"]

        generation_instruction = f"Tạo cuộc hội thoại tự nhiên kết hợp {len(chunks)} chủ đề khác nhau. Người dùng có thể chuyển từ chủ đề này sang chủ đề khác. Đặc biệt bao gồm các intent như greeting, unclear, human_request để cuộc hội thoại thực tế hơn"

        prompt = INTENT_CLASSIFICATION_CONVERSATION_GENERATION_PROMPT.format(
            num_conversations=num_conversations,
            content=combined_content,
            generation_instruction=generation_instruction,
        )

        response_json = self.generate_json_response(prompt)

        if response_json and "conversations" in response_json:
            conversations = response_json["conversations"]
            valid_conversations = []

            for i, conversation in enumerate(conversations):
                if "turns" in conversation and len(conversation["turns"]) >= 1:
                    valid_turns = []
                    for turn in conversation["turns"]:
                        if "user" in turn and "intent" in turn:
                            valid_turns.append(turn)

                    if valid_turns:
                        conv_obj = {
                            "id": f"multi_{'-'.join([c['id'] for c in chunks])}_{i}",
                            "turns": valid_turns,
                            "generation_method": "multi_chunk",
                            "source_chunks": [c["id"] for c in chunks],
                            "chunk_metadata": [c["metadata"] for c in chunks],
                        }
                        valid_conversations.append(conv_obj)

            print(
                f"✅ Generated {len(valid_conversations)} conversations for multi-chunk {[c['id'] for c in chunks]}"
            )
            return valid_conversations
        else:
            print(
                f"⚠️ No valid conversations generated for chunks {[c['id'] for c in chunks]}"
            )
            return []

    def create_intent_dataset(
        self,
        num_conversations_per_chunk: int = 3,
        save_path: str = "evaluation_data/datasets/intent_classification/viettelpay_intent_dataset.json",
    ) -> Dict:
        """Create intent classification dataset using two strategies only"""
        print(f"\n🚀 Creating intent classification dataset...")
        print(f"   Conversations per chunk: {num_conversations_per_chunk}")

        # Step 1: Get all chunks
        all_chunks = self.get_all_chunks()
        if not all_chunks:
            raise ValueError("No chunks found in knowledge base!")

        total_chunks = len(all_chunks)
        print(f"✅ Using all {total_chunks} chunks and shuffle them")
        random.shuffle(all_chunks)

        # Step 2: Split chunks for two strategies (60% single, 40% multi)
        split_point = int(total_chunks * 0.6)
        single_chunks = all_chunks[:split_point]
        multi_chunks = all_chunks[split_point:]

        print(f"📊 Distribution plan:")
        print(
            f"   • Single chunk: {len(single_chunks)} chunks → ~{len(single_chunks) * num_conversations_per_chunk} conversations"
        )
        print(
            f"   • Multi chunk: {len(multi_chunks)} chunks → ~{len(multi_chunks) // 2.5 * num_conversations_per_chunk} conversations"
        )

        all_conversations = []

        # Step 3: Generate single-chunk conversations
        print(f"\n💬 Generating single-chunk conversations...")
        for chunk in tqdm(single_chunks, desc="Single-chunk conversations"):
            conversations = self.generate_single_chunk_conversations(
                chunk, num_conversations_per_chunk
            )
            all_conversations.extend(conversations)
            time.sleep(0.1)

        # Step 4: Generate multi-chunk conversations (2-3 chunks randomly)
        print(f"\n🔀 Generating multi-chunk conversations...")
        random.shuffle(multi_chunks)  # Randomize order

        i = 0
        while i < len(multi_chunks):
            # Randomly choose to use 2 or 3 chunks
            chunk_count = random.choice([2, 3])
            chunk_group = multi_chunks[i : i + chunk_count]

            # Only proceed if we have at least 2 chunks
            if len(chunk_group) >= 2:
                conversations = self.generate_multi_chunk_conversations(
                    chunk_group, num_conversations_per_chunk
                )
                all_conversations.extend(conversations)
                time.sleep(0.1)

            i += chunk_count

        # Step 5: Track generation methods and intent distribution
        method_stats = defaultdict(int)
        intent_counts = Counter()

        for conv in all_conversations:
            method_stats[conv["generation_method"]] += 1
            for turn in conv["turns"]:
                intent_counts[turn["intent"]] += 1

        # Step 6: Populate dataset structure
        self.dataset["conversations"] = {conv["id"]: conv for conv in all_conversations}

        self.dataset["generation_methods"] = dict(method_stats)
        self.dataset["intent_distribution"] = dict(intent_counts)

        # Step 7: Update metadata
        total_user_messages = sum(len(conv["turns"]) for conv in all_conversations)

        self.dataset["metadata"].update(
            {
                "total_conversations": len(all_conversations),
                "total_user_messages": total_user_messages,
                "chunks_used": total_chunks,
                "conversations_per_chunk": num_conversations_per_chunk,
                "generation_distribution": dict(method_stats),
                "completion_timestamp": time.time(),
            }
        )

        # Step 8: Save dataset
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Intent classification dataset created successfully!")
        print(f"   📁 Saved to: {save_path}")
        print(f"   📊 Statistics:")
        print(f"      • Total conversations: {len(all_conversations)}")
        print(f"      • Total user messages: {total_user_messages}")
        print(f"      • Conversations per chunk: {num_conversations_per_chunk}")
        print(f"      • Generation methods: {dict(method_stats)}")
        print(f"      • Intent distribution: {dict(intent_counts)}")

        return self.dataset


class IntentClassificationEvaluator:
    """Evaluator for intent classification performance with method-specific analysis"""

    def __init__(self, dataset: Dict, llm_client):
        """Initialize evaluator with dataset and LLM client"""
        self.dataset = dataset
        self.llm_client = llm_client

        # Define expected intents
        self.expected_intents = [
            "greeting",
            "faq",
            "error_help",
            "procedure_guide",
            "human_request",
            "out_of_scope",
            "unclear",
        ]

        # Critical intents for business
        self.critical_intents = ["error_help", "human_request"]

        # Define flow mappings based on agent routing logic
        self.script_based_intents = {
            "greeting",
            "out_of_scope",
            "human_request",
            "unclear",
        }
        self.knowledge_based_intents = {
            "faq",
            "error_help",
            "procedure_guide",
        }

    def _get_intent_flow(self, intent: str) -> str:
        """Classify intent into flow type based on agent routing logic"""
        if intent in self.script_based_intents:
            return "script_based"
        elif intent in self.knowledge_based_intents:
            return "knowledge_based"
        else:
            return "unknown"

    def _make_json_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        try:
            import numpy as np

            if isinstance(obj, dict):
                return {k: self._make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [self._make_json_serializable(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        except ImportError:
            # If numpy is not available, just return the object as-is
            if isinstance(obj, dict):
                return {k: self._make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [self._make_json_serializable(item) for item in obj]
            else:
                return obj

    def calculate_essential_metrics(
        self, ground_truth: List[str], predictions: List[str]
    ) -> Dict:
        """Calculate only essential metrics: accuracy, macro, per-class"""
        try:
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support

            overall_accuracy = accuracy_score(ground_truth, predictions)

            # Calculate macro metrics (equal weight per intent)
            precision, recall, f1, support = precision_recall_fscore_support(
                ground_truth, predictions, average="macro", zero_division=0
            )

            macro_metrics = {
                "macro_precision": precision,
                "macro_recall": recall,
                "macro_f1": f1,
            }

            # Calculate per-class metrics
            precision_per_class, recall_per_class, f1_per_class, support_per_class = (
                precision_recall_fscore_support(
                    ground_truth, predictions, average=None, zero_division=0
                )
            )

            # Get unique labels
            unique_labels = sorted(list(set(ground_truth + predictions)))

            per_class_metrics = {}
            for i, label in enumerate(unique_labels):
                if i < len(precision_per_class):
                    per_class_metrics[label] = {
                        "precision": float(precision_per_class[i]),
                        "recall": float(recall_per_class[i]),
                        "f1": float(f1_per_class[i]),
                        "support": int(
                            support_per_class[i] if i < len(support_per_class) else 0
                        ),
                    }

            # Calculate critical intent recall
            critical_recall = {}
            for intent in self.critical_intents:
                if intent in per_class_metrics:
                    critical_recall[intent] = per_class_metrics[intent]["recall"]

            return {
                "overall_accuracy": float(overall_accuracy),
                "macro_precision": float(macro_metrics["macro_precision"]),
                "macro_recall": float(macro_metrics["macro_recall"]),
                "macro_f1": float(macro_metrics["macro_f1"]),
                "per_class_metrics": per_class_metrics,
                "critical_intent_recall": {
                    k: float(v) for k, v in critical_recall.items()
                },
            }

        except ImportError:
            print("⚠️ scikit-learn not installed. Using basic accuracy only.")
            overall_accuracy = sum(
                1 for gt, pred in zip(ground_truth, predictions) if gt == pred
            ) / len(predictions)

            return {"overall_accuracy": float(overall_accuracy)}

    def evaluate_intent_classification(self) -> Dict:
        """Evaluate intent classification performance with method and flow breakdown"""
        print(f"\n🎯 Running intent classification evaluation...")

        conversations = self.dataset["conversations"]

        # Initialize tracking
        all_predictions = []
        all_ground_truth = []
        method_results = defaultdict(lambda: {"predictions": [], "ground_truth": []})
        flow_results = defaultdict(lambda: {"predictions": [], "ground_truth": []})
        conversation_results = {}

        # Process each conversation
        for conv_id, conv_data in tqdm(
            conversations.items(), desc="Evaluating conversations"
        ):
            generation_method = conv_data.get("generation_method", "unknown")

            conversation_results[conv_id] = {
                "turns": [],
                "accuracy": 0,
                "generation_method": generation_method,
            }

            correct_predictions = 0
            total_turns = len(conv_data["turns"])

            # Process each turn in the conversation
            for turn_idx, turn in enumerate(conv_data["turns"]):
                user_message = turn["user"]
                ground_truth_intent = turn["intent"]

                try:
                    # Create messages in the format expected by classify_intent_node
                    messages = [HumanMessage(content=user_message)]

                    # Create a mock state for the intent classification node
                    state = ViettelPayState(messages=messages)

                    # Use the classify_intent_node directly
                    result_state = classify_intent_node(state, self.llm_client)
                    predicted_intent = result_state.get("intent", "unclear")

                    # Track results
                    is_correct = predicted_intent == ground_truth_intent
                    if is_correct:
                        correct_predictions += 1

                    # Add to overall tracking
                    all_predictions.append(predicted_intent)
                    all_ground_truth.append(ground_truth_intent)

                    # Add to method-specific tracking
                    method_results[generation_method]["predictions"].append(
                        predicted_intent
                    )
                    method_results[generation_method]["ground_truth"].append(
                        ground_truth_intent
                    )

                    # Add to flow-specific tracking
                    ground_truth_flow = self._get_intent_flow(ground_truth_intent)
                    predicted_flow = self._get_intent_flow(predicted_intent)

                    flow_results[ground_truth_flow]["predictions"].append(
                        predicted_intent
                    )
                    flow_results[ground_truth_flow]["ground_truth"].append(
                        ground_truth_intent
                    )

                    conversation_results[conv_id]["turns"].append(
                        {
                            "turn": turn_idx + 1,
                            "user_message": user_message,
                            "ground_truth": ground_truth_intent,
                            "predicted": predicted_intent,
                            "correct": is_correct,
                        }
                    )

                except Exception as e:
                    print(f"⚠️ Error processing turn {turn_idx} in {conv_id}: {e}")
                    # Use "unclear" as fallback prediction
                    all_predictions.append("unclear")
                    all_ground_truth.append(ground_truth_intent)
                    method_results[generation_method]["predictions"].append("unclear")
                    method_results[generation_method]["ground_truth"].append(
                        ground_truth_intent
                    )

                    # Add to flow-specific tracking (for errors)
                    ground_truth_flow = self._get_intent_flow(ground_truth_intent)
                    flow_results[ground_truth_flow]["predictions"].append("unclear")
                    flow_results[ground_truth_flow]["ground_truth"].append(
                        ground_truth_intent
                    )

            # Calculate conversation accuracy
            conversation_results[conv_id]["accuracy"] = float(
                correct_predictions / total_turns if total_turns > 0 else 0
            )

        # Calculate overall metrics
        overall_metrics = self.calculate_essential_metrics(
            all_ground_truth, all_predictions
        )

        # Calculate method-specific metrics
        method_metrics = {}
        for method, method_data in method_results.items():
            if method_data["predictions"]:  # Ensure we have data
                method_metrics[method] = self.calculate_essential_metrics(
                    method_data["ground_truth"], method_data["predictions"]
                )
                method_metrics[method]["total_messages"] = len(
                    method_data["predictions"]
                )

        # Calculate flow-specific metrics
        flow_metrics = {}
        for flow, flow_data in flow_results.items():
            if flow_data["predictions"]:  # Ensure we have data
                flow_metrics[flow] = self.calculate_essential_metrics(
                    flow_data["ground_truth"], flow_data["predictions"]
                )
                flow_metrics[flow]["total_messages"] = len(flow_data["predictions"])

        results = {
            "overall_metrics": overall_metrics,
            "method_specific_metrics": method_metrics,
            "flow_specific_metrics": flow_metrics,
            "conversation_results": conversation_results,
            "intent_distribution": {
                "ground_truth": dict(Counter(all_ground_truth)),
                "predicted": dict(Counter(all_predictions)),
            },
            "generation_methods": self.dataset.get("generation_methods", {}),
        }

        # Make sure all values are JSON serializable
        results = self._make_json_serializable(results)

        return results

    def print_evaluation_results(self, results: Dict):
        """Print comprehensive evaluation results"""
        print(f"\n🎯 INTENT CLASSIFICATION EVALUATION RESULTS")
        print("=" * 60)

        # Overall performance
        overall = results["overall_metrics"]
        print(f"\n📊 Overall Performance:")
        print(f"   Accuracy: {overall['overall_accuracy']:.3f}")
        if "macro_precision" in overall:
            print(f"   Macro Precision: {overall['macro_precision']:.3f}")
            print(f"   Macro Recall: {overall['macro_recall']:.3f}")
            print(f"   Macro F1: {overall['macro_f1']:.3f}")

        # Per-class performance
        if "per_class_metrics" in overall:
            print(f"\n📋 Per-Class Performance:")
            print(
                f"{'Intent':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}"
            )
            print("-" * 65)

            per_class = overall["per_class_metrics"]
            for intent in self.expected_intents:
                if intent in per_class:
                    metrics = per_class[intent]
                    print(
                        f"{intent:<15} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1']:<10.3f} {metrics['support']:<10}"
                    )

        # Critical intents performance
        if "critical_intent_recall" in overall:
            print(f"\n🚨 Critical Intent Performance:")
            for intent, recall in overall["critical_intent_recall"].items():
                status = "✅" if recall >= 0.85 else "⚠️" if recall >= 0.75 else "❌"
                print(f"   {status} {intent}: Recall = {recall:.3f}")

        # Method-specific performance
        print(f"\n🔄 Performance by Generation Method:")
        method_metrics = results["method_specific_metrics"]
        if method_metrics:
            print(f"{'Method':<20} {'Accuracy':<10} {'Macro F1':<10} {'Messages':<10}")
            print("-" * 55)

            for method, metrics in method_metrics.items():
                accuracy = metrics["overall_accuracy"]
                macro_f1 = metrics.get("macro_f1", 0)
                total_msgs = metrics["total_messages"]
                print(
                    f"{method:<20} {accuracy:<10.3f} {macro_f1:<10.3f} {total_msgs:<10}"
                )

        # Flow-specific performance
        print(f"\n🔀 Performance by Agent Flow:")
        flow_metrics = results["flow_specific_metrics"]
        if flow_metrics:
            print(
                f"{'Flow Type':<20} {'Accuracy':<10} {'Macro F1':<10} {'Messages':<10}"
            )
            print("-" * 55)

            for flow, metrics in flow_metrics.items():
                accuracy = metrics["overall_accuracy"]
                macro_f1 = metrics.get("macro_f1", 0)
                total_msgs = metrics["total_messages"]
                flow_display = f"{flow}_flow"
                print(
                    f"{flow_display:<20} {accuracy:<10.3f} {macro_f1:<10.3f} {total_msgs:<10}"
                )

        # Intent distribution comparison
        print(f"\n📈 Intent Distribution:")
        gt_dist = results["intent_distribution"]["ground_truth"]
        pred_dist = results["intent_distribution"]["predicted"]

        print(f"{'Intent':<15} {'Ground Truth':<15} {'Predicted':<15}")
        print("-" * 50)

        all_intents = set(list(gt_dist.keys()) + list(pred_dist.keys()))
        for intent in sorted(all_intents):
            gt_count = gt_dist.get(intent, 0)
            pred_count = pred_dist.get(intent, 0)
            print(f"{intent:<15} {gt_count:<15} {pred_count:<15}")

        # Method insights
        print(f"\n💡 Method-Specific Insights:")
        if method_metrics:
            method_accuracies = {
                method: metrics["overall_accuracy"]
                for method, metrics in method_metrics.items()
            }
            best_method = max(
                method_accuracies.keys(), key=lambda k: method_accuracies[k]
            )
            worst_method = min(
                method_accuracies.keys(), key=lambda k: method_accuracies[k]
            )

            print(
                f"   • Best performing method: {best_method} ({method_accuracies[best_method]:.3f})"
            )
            print(
                f"   • Most challenging method: {worst_method} ({method_accuracies[worst_method]:.3f})"
            )
            print(
                f"   • Performance gap: {method_accuracies[best_method] - method_accuracies[worst_method]:.3f}"
            )

        # Flow insights
        print(f"\n🔀 Flow-Specific Insights:")
        if flow_metrics:
            flow_accuracies = {
                flow: metrics["overall_accuracy"]
                for flow, metrics in flow_metrics.items()
            }

            if len(flow_accuracies) >= 2:
                best_flow = max(
                    flow_accuracies.keys(), key=lambda k: flow_accuracies[k]
                )
                worst_flow = min(
                    flow_accuracies.keys(), key=lambda k: flow_accuracies[k]
                )

                print(
                    f"   • Best performing flow: {best_flow} ({flow_accuracies[best_flow]:.3f})"
                )
                print(
                    f"   • Most challenging flow: {worst_flow} ({flow_accuracies[worst_flow]:.3f})"
                )
                print(
                    f"   • Flow performance gap: {flow_accuracies[best_flow] - flow_accuracies[worst_flow]:.3f}"
                )

                # Provide interpretation
                if (
                    "script_based" in flow_accuracies
                    and "knowledge_based" in flow_accuracies
                ):
                    script_acc = flow_accuracies["script_based"]
                    kb_acc = flow_accuracies["knowledge_based"]

                    if script_acc > kb_acc:
                        print(
                            f"   • Script-based intents are easier to classify ({script_acc:.3f} vs {kb_acc:.3f})"
                        )
                    elif kb_acc > script_acc:
                        print(
                            f"   • Knowledge-based intents are easier to classify ({kb_acc:.3f} vs {script_acc:.3f})"
                        )
                    else:
                        print(
                            f"   • Both flows perform similarly ({script_acc:.3f} vs {kb_acc:.3f})"
                        )
            else:
                for flow, accuracy in flow_accuracies.items():
                    print(f"   • {flow} flow accuracy: {accuracy:.3f}")

        # Success criteria check
        print(f"\n✅ Success Criteria Check:")
        accuracy = overall["overall_accuracy"]
        if accuracy >= 0.80:
            print(f"   🎉 GOOD: Overall accuracy {accuracy:.3f} >= 0.80")
        elif accuracy >= 0.75:
            print(f"   ⚠️ OKAY: Overall accuracy {accuracy:.3f} >= 0.75")
        else:
            print(f"   ❌ NEEDS WORK: Overall accuracy {accuracy:.3f} < 0.75")


def main():
    """Main function for simplified intent classification evaluation"""
    parser = argparse.ArgumentParser(
        description="Simplified ViettelPay Intent Classification Evaluation"
    )
    parser.add_argument(
        "--mode",
        choices=["create", "evaluate", "full"],
        default="full",
        help="Mode: create dataset, evaluate, or full pipeline",
    )
    parser.add_argument(
        "--dataset-path",
        default="evaluation_data/datasets/intent_classification/viettelpay_intent_dataset.json",
        help="Path to intent dataset",
    )
    parser.add_argument(
        "--results-path",
        default="evaluation_data/results/intent_classification/viettelpay_intent_results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--conversations-per-chunk",
        type=int,
        default=3,
        help="Number of conversations per chunk (default: 3)",
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
        # Initialize components based on mode
        kb = None
        if args.mode in ["create", "full"]:
            # Initialize knowledge base only if creating dataset
            print("🔧 Initializing ViettelPay knowledge base...")
            kb = ViettelKnowledgeBase()
            if not kb.load_knowledge_base(args.knowledge_base_path):
                print(
                    "❌ Failed to load knowledge base. Please run build_database_script.py first."
                )
                return

        # Step 1: Create dataset if requested
        if args.mode in ["create", "full"]:
            print(f"\n🎯 Creating simplified intent classification dataset...")
            creator = IntentDatasetCreator(GOOGLE_API_KEY, kb)

            dataset = creator.create_intent_dataset(
                num_conversations_per_chunk=args.conversations_per_chunk,
                save_path=args.dataset_path,
            )

        # Step 2: Evaluate if requested
        if args.mode in ["evaluate", "full"]:
            print(f"\n📊 Evaluating intent classification...")

            # Load dataset if not created in this run
            if args.mode == "evaluate":
                if not os.path.exists(args.dataset_path):
                    print(f"❌ Dataset not found: {args.dataset_path}")
                    return

                with open(args.dataset_path, "r", encoding="utf-8") as f:
                    dataset = json.load(f)

            # Initialize LLM client for intent classification
            print("🤖 Initializing LLM client for intent classification...")
            llm_client = LLMClientFactory.create_client(
                "gemini", api_key=GOOGLE_API_KEY, model="gemini-2.0-flash"
            )

            # Run evaluation
            evaluator = IntentClassificationEvaluator(dataset, llm_client)
            results = evaluator.evaluate_intent_classification()
            evaluator.print_evaluation_results(results)

            # Save results
            if args.results_path:
                with open(args.results_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"\n💾 Results saved to: {args.results_path}")

        print(f"\n✅ Intent classification evaluation completed successfully!")
        print(f"\n💡 Summary improvements made:")
        print(f"   • Removed pattern-based generation for simplicity")
        print(f"   • Added configurable conversations-per-chunk (default: 3)")
        print(f"   • Improved chunk mixing (random 2-3 chunks)")
        print(f"   • Enhanced prompts to include non-topic intents")
        print(f"   • Added flow-specific analysis (script-based vs knowledge-based)")

    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
