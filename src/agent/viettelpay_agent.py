"""
ViettelPay AI Agent using LangGraph
Multi-turn conversation support with short-term memory using InMemorySaver
"""

import os
from typing import Dict, Optional
from functools import partial
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage

from src.agent.nodes import (
    ViettelPayState,
    classify_intent_node,
    query_enhancement_node,
    knowledge_retrieval_node,
    script_response_node,
    generate_response_node,
    route_after_intent_classification,
    route_after_query_enhancement,
    route_after_knowledge_retrieval,
)

# Load environment variables from .env file
load_dotenv()


class ViettelPayAgent:
    """Main ViettelPay AI Agent using LangGraph workflow with multi-turn conversation support"""

    def __init__(
        self,
        knowledge_base_path: str = "./knowledge_base",
        scripts_file: Optional[str] = None,
        llm_provider: str = "gemini",
    ):
        self.knowledge_base_path = knowledge_base_path
        self.scripts_file = scripts_file or "./viettelpay_docs/processed/kich_ban.csv"
        self.llm_provider = llm_provider

        # Initialize LLM client once during agent creation
        print(f"🧠 Initializing LLM client ({llm_provider})...")
        from src.llm.llm_client import LLMClientFactory

        self.llm_client = LLMClientFactory.create_client(llm_provider)
        print(f"✅ LLM client initialized and ready")

        # Initialize knowledge retriever once during agent creation
        print(f"📚 Initializing knowledge retriever...")
        try:
            from src.knowledge_base.viettel_knowledge_base import ViettelKnowledgeBase

            self.knowledge_base = ViettelKnowledgeBase()
            ensemble_retriever = self.knowledge_base.load_knowledge_base(
                knowledge_base_path
            )
            if not ensemble_retriever:
                raise ValueError(
                    f"Knowledge base not found at {knowledge_base_path}. Run build_database_script.py first."
                )
            print(f"✅ Knowledge retriever initialized and ready")
        except Exception as e:
            print(f"⚠️ Knowledge retriever initialization failed: {e}")
            self.knowledge_base = None

        # Initialize checkpointer for short-term memory
        self.checkpointer = InMemorySaver()

        # Build workflow with pre-initialized components
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

        print("✅ ViettelPay Agent initialized with multi-turn conversation support")

    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow with pre-initialized components"""

        # Create workflow graph
        workflow = StateGraph(ViettelPayState)

        # Create node functions with pre-bound components using functools.partial
        # This eliminates the need to initialize components in each node call
        classify_intent_with_llm = partial(
            classify_intent_node, llm_client=self.llm_client
        )
        query_enhancement_with_llm = partial(
            query_enhancement_node, llm_client=self.llm_client
        )
        knowledge_retrieval_with_retriever = partial(
            knowledge_retrieval_node, knowledge_retriever=self.knowledge_base
        )
        generate_response_with_llm = partial(
            generate_response_node, llm_client=self.llm_client
        )

        # Add nodes (some with pre-bound components, some without)
        workflow.add_node("classify_intent", classify_intent_with_llm)
        workflow.add_node("query_enhancement", query_enhancement_with_llm)
        workflow.add_node("knowledge_retrieval", knowledge_retrieval_with_retriever)
        workflow.add_node(
            "script_response", script_response_node
        )  # No pre-bound components needed
        workflow.add_node("generate_response", generate_response_with_llm)

        # Set entry point
        workflow.set_entry_point("classify_intent")

        # Add conditional routing after intent classification
        workflow.add_conditional_edges(
            "classify_intent",
            route_after_intent_classification,
            {
                "script_response": "script_response",
                "query_enhancement": "query_enhancement",
            },
        )

        # Script responses go directly to end
        workflow.add_edge("script_response", END)

        # Query enhancement goes to knowledge retrieval
        workflow.add_edge("query_enhancement", "knowledge_retrieval")

        # Knowledge retrieval goes to response generation
        workflow.add_edge("knowledge_retrieval", "generate_response")
        workflow.add_edge("generate_response", END)

        print("🔄 LangGraph workflow built successfully with optimized component usage")
        return workflow

    def process_message(self, user_message: str, thread_id: str = "default") -> Dict:
        """Process a user message in a multi-turn conversation"""

        print(f"\n💬 Processing message: '{user_message}' (thread: {thread_id})")
        print("=" * 50)

        # Create configuration with thread_id for conversation memory
        config = {"configurable": {"thread_id": thread_id}}

        try:
            # Create human message
            human_message = HumanMessage(content=user_message)

            # Initialize state with the new message
            initial_state = {
                "messages": [human_message],
                "intent": None,
                "confidence": None,
                "enhanced_query": None,
                "retrieved_docs": None,
                "response_type": None,
                "error": None,
                "processing_info": None,
            }

            # Run workflow with memory
            result = self.app.invoke(initial_state, config)

            # Extract response from the last AI message
            messages = result.get("messages", [])
            if messages:
                # Get the last AI message
                last_message = messages[-1]
                if hasattr(last_message, "content"):
                    response = last_message.content
                else:
                    response = str(last_message)
            else:
                response = "Xin lỗi, em không thể xử lý yêu cầu này."

            response_type = result.get("response_type", "unknown")
            intent = result.get("intent", "unknown")
            confidence = result.get("confidence", 0.0)
            enhanced_query = result.get("enhanced_query", "")
            error = result.get("error")

            # Build response info
            response_info = {
                "response": response,
                "intent": intent,
                "confidence": confidence,
                "response_type": response_type,
                "enhanced_query": enhanced_query,
                "success": error is None,
                "error": error,
                "thread_id": thread_id,
                "message_count": len(messages),
            }

            print(f"✅ Response generated successfully")
            print(f"   Intent: {intent} (confidence: {confidence})")
            print(f"   Type: {response_type}")
            if enhanced_query and enhanced_query != user_message:
                print(f"   Enhanced query: {enhanced_query}")
            print(f"   Thread: {thread_id}")

            return response_info

        except Exception as e:
            print(f"❌ Workflow error: {e}")

            return {
                "response": "Xin lỗi, em gặp lỗi kỹ thuật. Vui lòng thử lại sau.",
                "intent": "error",
                "confidence": 0.0,
                "response_type": "error",
                "enhanced_query": "",
                "success": False,
                "error": str(e),
                "thread_id": thread_id,
                "message_count": 0,
            }

    def chat(self, user_message: str, thread_id: str = "default") -> str:
        """Simple chat interface - returns just the response text"""
        result = self.process_message(user_message, thread_id)
        return result["response"]

    def get_conversation_history(self, thread_id: str = "default") -> list:
        """Get conversation history for a specific thread"""
        try:
            config = {"configurable": {"thread_id": thread_id}}

            # Get the current state to access message history
            current_state = self.app.get_state(config)

            if current_state and current_state.values.get("messages"):
                messages = current_state.values["messages"]
                history = []

                for msg in messages:
                    if hasattr(msg, "type") and hasattr(msg, "content"):
                        role = "user" if msg.type == "human" else "assistant"
                        history.append({"role": role, "content": msg.content})
                    elif hasattr(msg, "role") and hasattr(msg, "content"):
                        history.append({"role": msg.role, "content": msg.content})

                return history
            else:
                return []

        except Exception as e:
            print(f"❌ Error getting conversation history: {e}")
            return []

    def clear_conversation(self, thread_id: str = "default") -> bool:
        """Clear conversation history for a specific thread"""
        try:
            # Note: InMemorySaver doesn't have a direct clear method
            # The conversation will be cleared when the app is restarted
            # For persistent memory, you'd need to implement a clear method
            print(f"📝 Conversation clearing requested for thread: {thread_id}")
            print("   Note: InMemorySaver conversations clear on app restart")
            return True
        except Exception as e:
            print(f"❌ Error clearing conversation: {e}")
            return False

    def get_workflow_info(self) -> Dict:
        """Get information about the workflow structure"""
        return {
            "nodes": [
                "classify_intent",
                "query_enhancement",
                "knowledge_retrieval",
                "script_response",
                "generate_response",
            ],
            "entry_point": "classify_intent",
            "knowledge_base_path": self.knowledge_base_path,
            "scripts_file": self.scripts_file,
            "llm_provider": self.llm_provider,
            "memory_type": "InMemorySaver",
            "multi_turn": True,
            "query_enhancement": True,
            "optimizations": {
                "llm_client": "Single initialization with functools.partial",
                "knowledge_retriever": "Single initialization with functools.partial",
            },
        }

    def health_check(self) -> Dict:
        """Check if all components are working"""

        health_status = {
            "agent": True,
            "workflow": True,
            "memory": True,
            "llm": False,
            "knowledge_base": False,
            "scripts": False,
            "overall": False,
        }

        try:
            # Test LLM client (already initialized)
            test_response = self.llm_client.generate("Hello", temperature=0.1)
            health_status["llm"] = bool(test_response)
            print("✅ LLM client working")

        except Exception as e:
            print(f"⚠️ LLM health check failed: {e}")
            health_status["llm"] = False

        try:
            # Test memory/checkpointer
            test_config = {"configurable": {"thread_id": "health_check"}}
            test_state = {"messages": [HumanMessage(content="test")]}

            # Try to invoke with memory
            self.app.invoke(test_state, test_config)
            health_status["memory"] = True
            print("✅ Memory/checkpointer working")

        except Exception as e:
            print(f"⚠️ Memory health check failed: {e}")
            health_status["memory"] = False

        try:
            # Test knowledge base (using pre-initialized retriever)
            if self.knowledge_base:
                # Test a simple search to verify it's working
                test_docs = self.knowledge_base.search("test", k=1)
                health_status["knowledge_base"] = True
                print("✅ Knowledge retriever working")
            else:
                health_status["knowledge_base"] = False
                print("❌ Knowledge retriever not initialized")

        except Exception as e:
            print(f"⚠️ Knowledge base health check failed: {e}")
            health_status["knowledge_base"] = False

        try:
            # Test scripts
            from src.agent.scripts import ConversationScripts

            scripts = ConversationScripts(self.scripts_file)
            health_status["scripts"] = len(scripts.get_all_script_types()) > 0

        except Exception as e:
            print(f"⚠️ Scripts health check failed: {e}")

        # Overall health
        health_status["overall"] = all(
            [
                health_status["agent"],
                health_status["memory"],
                health_status["llm"],
                health_status["knowledge_base"],
                health_status["scripts"],
            ]
        )

        return health_status


# Usage example and testing
if __name__ == "__main__":
    # Initialize agent
    agent = ViettelPayAgent()

    # Health check
    print("\n🏥 Health Check:")
    health = agent.health_check()
    for component, status in health.items():
        status_icon = "✅" if status else "❌"
        print(f"   {component}: {status_icon}")

    if not health["overall"]:
        print("\n⚠️ Some components are not healthy. Check requirements and data files.")
        exit(1)

    print(f"\n🤖 Agent ready! Workflow info: {agent.get_workflow_info()}")

    # Test multi-turn conversation with query enhancement
    test_thread = "test_conversation"

    print(
        f"\n🧪 Testing multi-turn conversation with query enhancement (thread: {test_thread}):"
    )

    test_messages = [
        "Xin chào!",
        "Mã lỗi 606 là gì?",
        "Làm sao khắc phục?",  # This should be enhanced to "làm sao khắc phục lỗi 606"
        "Còn lỗi nào khác tương tự không?",  # This should be enhanced with error context
        "Cảm ơn bạn!",
    ]

    for i, message in enumerate(test_messages, 1):
        print(f"\n--- Turn {i} ---")
        result = agent.process_message(message, test_thread)
        print(f"User: {message}")
        print(f"Bot: {result['response'][:150]}...")

        if result.get("enhanced_query") and result["enhanced_query"] != message:
            print(f"🚀 Query enhanced: {result['enhanced_query']}")

        # Show conversation history
        if i > 1:
            history = agent.get_conversation_history(test_thread)
            print(f"History length: {len(history)} messages")

    print(f"\n📜 Final conversation history:")
    history = agent.get_conversation_history(test_thread)
    for i, msg in enumerate(history, 1):
        print(f"  {i}. {msg['role']}: {msg['content'][:100]}...")
