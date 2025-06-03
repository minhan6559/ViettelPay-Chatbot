"""
ViettelPay AI Agent - Multi-turn Conversation Streamlit Interface
"""

import streamlit as st
from dotenv import load_dotenv
import os
import uuid

# Load environment variables
load_dotenv()

# Simple page config
st.set_page_config(
    page_title="ViettelPay AI Assistant", page_icon="💳", layout="centered"
)

# Header
st.title("💳 ViettelPay AI Assistant")
st.caption("Trợ lý ảo hỗ trợ dịch vụ ViettelPay Pro - Multi-turn Conversation")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

if "thread_id" not in st.session_state:
    # Generate a unique thread ID for this session
    st.session_state.thread_id = str(uuid.uuid4())[:8]

if "conversation_started" not in st.session_state:
    st.session_state.conversation_started = False

# Sidebar
with st.sidebar:
    st.header("🛠️ Controls")

    # Initialize agent
    if st.session_state.agent is None:
        if st.button("🚀 Initialize Agent"):
            try:
                with st.spinner("Loading..."):
                    from src.agent.viettelpay_agent import ViettelPayAgent

                    st.session_state.agent = ViettelPayAgent()
                    st.success("✅ Agent ready!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.success("✅ Agent ready!")

        # Health check
        if st.button("🏥 Health Check"):
            try:
                health = st.session_state.agent.health_check()
                for component, status in health.items():
                    if component != "overall":
                        icon = "✅" if status else "❌"
                        st.write(f"{icon} {component}")
            except Exception as e:
                st.error(f"Health check failed: {e}")

    st.divider()

    # Conversation Management
    st.subheader("💬 Conversation")

    # Show current thread ID
    st.write(f"**Thread ID:** `{st.session_state.thread_id}`")

    # Show conversation stats
    if st.session_state.agent and st.session_state.conversation_started:
        try:
            history = st.session_state.agent.get_conversation_history(
                st.session_state.thread_id
            )
            st.write(f"**Messages:** {len(history)}")
        except:
            st.write("**Messages:** Unable to count")

    # Clear current conversation
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_started = False
        if st.session_state.agent:
            st.session_state.agent.clear_conversation(st.session_state.thread_id)
        st.rerun()

    # Start new conversation
    if st.button("🆕 New Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.conversation_started = False
        st.rerun()

    st.divider()

    # Conversation History Viewer
    if st.session_state.agent and st.session_state.conversation_started:
        st.subheader("📜 History")
        if st.button("🔍 Show Full History"):
            try:
                history = st.session_state.agent.get_conversation_history(
                    st.session_state.thread_id
                )
                with st.expander("Full Conversation History", expanded=True):
                    for i, msg in enumerate(history, 1):
                        role_icon = "👤" if msg["role"] == "user" else "🤖"
                        st.write(
                            f"{i}. {role_icon} **{msg['role'].title()}:** {msg['content']}"
                        )
            except Exception as e:
                st.error(f"Error loading history: {e}")

# Main chat area
st.subheader("💬 Chat")

# Display messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.write(msg["content"])
            # Show metadata if available
            if "metadata" in msg:
                with st.expander("📊 Details"):
                    metadata = msg["metadata"]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Intent", metadata.get("intent", "unknown"))
                    with col2:
                        st.metric("Confidence", f"{metadata.get('confidence', 0):.2f}")
                    with col3:
                        st.metric("Type", metadata.get("response_type", "unknown"))

                    # Show enhanced query if available and different from original
                    if metadata.get("enhanced_query"):
                        st.write(f"**🚀 Enhanced Query:** {metadata['enhanced_query']}")

                    if metadata.get("thread_id"):
                        st.write(f"**Thread:** {metadata['thread_id']}")
                    if metadata.get("message_count"):
                        st.write(f"**Total Messages:** {metadata['message_count']}")
    elif msg["role"] == "error":
        st.error(msg["content"])

# Chat input
if prompt := st.chat_input("Type your question..."):
    if st.session_state.agent is None:
        st.error("Please initialize the agent first!")
        st.stop()

    # Mark conversation as started
    st.session_state.conversation_started = True

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Process message
    try:
        with st.spinner("Processing..."):
            # Use the session's thread_id for conversation continuity
            result = st.session_state.agent.process_message(
                prompt, st.session_state.thread_id
            )

            # Add response
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["response"],
                    "metadata": {
                        "intent": result.get("intent"),
                        "confidence": result.get("confidence"),
                        "response_type": result.get("response_type"),
                        "enhanced_query": result.get("enhanced_query"),
                        "success": result.get("success"),
                        "thread_id": result.get("thread_id"),
                        "message_count": result.get("message_count"),
                    },
                }
            )

            if result.get("error"):
                st.session_state.messages.append(
                    {"role": "error", "content": f"Error: {result['error']}"}
                )

    except Exception as e:
        st.session_state.messages.append(
            {"role": "error", "content": f"System error: {str(e)}"}
        )

    st.rerun()

# Footer
st.caption(
    "🚀 Powered by LangGraph & Gemini AI | Multi-turn conversation with InMemorySaver"
)
