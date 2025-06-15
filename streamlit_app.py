"""
Trợ lý AI ViettelPay - Giao diện Streamlit cho cuộc trò chuyện đa lượt
"""

import streamlit as st
import os
import uuid
import time

# For backward compatibility with local development using .env files
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not required in production

# Simple page config
st.set_page_config(page_title="Trợ lý AI ViettelPay", page_icon="💳", layout="centered")

# Header
st.title("💳 Trợ lý AI ViettelPay")
st.caption("Trợ lý ảo hỗ trợ dịch vụ ViettelPay Pro - Cuộc trò chuyện đa lượt")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

if "agent_initialization_status" not in st.session_state:
    st.session_state.agent_initialization_status = "not_started"

if "thread_id" not in st.session_state:
    # Generate a unique thread ID for this session
    st.session_state.thread_id = str(uuid.uuid4())[:8]

if "conversation_started" not in st.session_state:
    st.session_state.conversation_started = False


@st.cache_resource
def initialize_agent():
    """Khởi tạo agent ViettelPay với bộ nhớ đệm"""
    try:
        print("[DEBUG] Starting agent initialization...")
        st.write("🔍 [DEBUG] Starting agent initialization...")

        from src.agent.viettelpay_agent import ViettelPayAgent

        print("[DEBUG] ViettelPayAgent imported successfully")
        st.write("🔍 [DEBUG] ViettelPayAgent imported successfully")

        print("[DEBUG] Creating ViettelPayAgent instance...")
        st.write("🔍 [DEBUG] Creating ViettelPayAgent instance...")
        agent = ViettelPayAgent()

        print("[DEBUG] ViettelPayAgent created successfully!")
        st.write("🔍 [DEBUG] ViettelPayAgent created successfully!")
        return agent

    except Exception as e:
        error_msg = f"❌ Lỗi khởi tạo agent: {str(e)}"
        print(f"[ERROR] {error_msg}")
        print(f"[ERROR] Exception type: {type(e).__name__}")
        print(f"[ERROR] Exception details: {repr(e)}")

        st.error(error_msg)
        st.error(f"🔍 [DEBUG] Exception type: {type(e).__name__}")
        st.error(f"🔍 [DEBUG] Exception details: {repr(e)}")

        # Try to show traceback
        import traceback

        traceback_str = traceback.format_exc()
        print(f"[ERROR] Full traceback:\n{traceback_str}")
        st.error(f"🔍 [DEBUG] Full traceback:")
        st.code(traceback_str)

        return None


# Auto-initialize agent when app starts
if (
    st.session_state.agent is None
    and st.session_state.agent_initialization_status == "not_started"
):
    st.session_state.agent_initialization_status = "initializing"

    with st.spinner("🚀 Đang khởi tạo Trợ lý AI ViettelPay..."):
        st.session_state.agent = initialize_agent()
        if st.session_state.agent:
            st.session_state.agent_initialization_status = "success"
            st.success("✅ Khởi tạo agent thành công!")
        else:
            st.session_state.agent_initialization_status = "failed"
            st.error("❌ Khởi tạo agent thất bại")

# Sidebar
with st.sidebar:
    st.header("🛠️ Điều khiển")

    # Agent status
    if st.session_state.agent_initialization_status == "initializing":
        st.info("🔄 Đang khởi tạo agent...")
    elif st.session_state.agent_initialization_status == "success":
        st.success("✅ Agent đã sẵn sàng!")

        # Health check
        if st.button("🏥 Kiểm tra tình trạng"):
            try:
                print("[DEBUG] Starting health check...")
                st.write("🔍 [DEBUG] Starting health check...")

                health = st.session_state.agent.health_check()
                print(f"[DEBUG] Health check results: {health}")

                for component, status in health.items():
                    if component != "overall":
                        icon = "✅" if status else "❌"
                        st.write(f"{icon} {component}")
                        print(f"[DEBUG] {component}: {status}")

            except Exception as e:
                error_msg = f"Kiểm tra tình trạng thất bại: {e}"
                print(f"[ERROR] {error_msg}")
                print(f"[ERROR] Health check exception: {repr(e)}")

                st.error(error_msg)
                st.error(f"🔍 [DEBUG] Exception: {repr(e)}")

                import traceback

                traceback_str = traceback.format_exc()
                print(f"[ERROR] Health check traceback:\n{traceback_str}")
                st.code(traceback_str)

    elif st.session_state.agent_initialization_status == "failed":
        st.error("❌ Khởi tạo agent thất bại")
        if st.button("🔄 Thử lại khởi tạo"):
            st.session_state.agent_initialization_status = "not_started"
            st.session_state.agent = None
            st.rerun()

    st.divider()

    # Conversation Management
    st.subheader("💬 Cuộc trò chuyện")

    # Show current thread ID
    st.write(f"**ID Cuộc trò chuyện:** `{st.session_state.thread_id}`")

    # Show conversation stats
    if st.session_state.agent and st.session_state.conversation_started:
        try:
            print(
                f"[DEBUG] Getting conversation history for thread: {st.session_state.thread_id}"
            )
            history = st.session_state.agent.get_conversation_history(
                st.session_state.thread_id
            )
            st.write(f"**Số tin nhắn:** {len(history)}")
            print(f"[DEBUG] Conversation history length: {len(history)}")
        except Exception as e:
            print(f"[DEBUG] Error getting conversation history: {e}")
            st.write("**Số tin nhắn:** Không thể đếm")

    # Clear current conversation
    if st.button("🗑️ Xóa cuộc trò chuyện"):
        st.session_state.messages = []
        st.session_state.conversation_started = False
        if st.session_state.agent:
            st.session_state.agent.clear_conversation(st.session_state.thread_id)
        st.rerun()

    # Start new conversation
    if st.button("🆕 Cuộc trò chuyện mới"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.conversation_started = False
        st.rerun()

    st.divider()

    # Conversation History Viewer
    if st.session_state.agent and st.session_state.conversation_started:
        st.subheader("📜 Lịch sử")
        if st.button("🔍 Xem toàn bộ lịch sử"):
            try:
                history = st.session_state.agent.get_conversation_history(
                    st.session_state.thread_id
                )
                with st.expander("Lịch sử cuộc trò chuyện đầy đủ", expanded=True):
                    for i, msg in enumerate(history, 1):
                        role_icon = "👤" if msg["role"] == "user" else "🤖"
                        role_text = "Người dùng" if msg["role"] == "user" else "Trợ lý"
                        st.write(f"{i}. {role_icon} **{role_text}:** {msg['content']}")
            except Exception as e:
                st.error(f"Lỗi tải lịch sử: {e}")

# Main chat area
st.subheader("💬 Trò chuyện")

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
                with st.expander("📊 Chi tiết"):
                    metadata = msg["metadata"]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Ý định", metadata.get("intent", "không xác định"))
                    with col2:
                        st.metric("Độ tin cậy", f"{metadata.get('confidence', 0):.2f}")
                    with col3:
                        st.metric(
                            "Loại", metadata.get("response_type", "không xác định")
                        )

                    # Show enhanced query if available and different from original
                    if metadata.get("enhanced_query"):
                        st.write(
                            f"**🚀 Câu hỏi được tối ưu:** {metadata['enhanced_query']}"
                        )

                    if metadata.get("thread_id"):
                        st.write(f"**Cuộc trò chuyện:** {metadata['thread_id']}")
                    if metadata.get("message_count"):
                        st.write(f"**Tổng số tin nhắn:** {metadata['message_count']}")
    elif msg["role"] == "error":
        st.error(msg["content"])

# Chat input
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    if st.session_state.agent is None:
        st.error("Vui lòng đợi quá trình khởi tạo agent hoàn thành!")
        st.stop()

    # Mark conversation as started
    st.session_state.conversation_started = True

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Process message
    try:
        with st.spinner("Đang xử lý..."):
            print(f"[DEBUG] Processing message: {prompt[:100]}...")
            print(f"[DEBUG] Thread ID: {st.session_state.thread_id}")

            # Use the session's thread_id for conversation continuity

            # Check the time processing
            start_time = time.time()
            print("[DEBUG] Calling agent.process_message()...")

            result = st.session_state.agent.process_message(
                prompt, st.session_state.thread_id
            )

            end_time = time.time()
            processing_time = end_time - start_time
            print(f"[DEBUG] Thời gian xử lý: {processing_time:.2f} giây")
            print(f"[DEBUG] Result keys: {list(result.keys()) if result else 'None'}")

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
                print(f"[DEBUG] Result contains error: {result['error']}")
                st.session_state.messages.append(
                    {"role": "error", "content": f"Lỗi: {result['error']}"}
                )

    except Exception as e:
        error_msg = f"Lỗi hệ thống: {str(e)}"
        print(f"[ERROR] {error_msg}")
        print(f"[ERROR] Message processing exception: {repr(e)}")

        # Add detailed error info
        import traceback

        traceback_str = traceback.format_exc()
        print(f"[ERROR] Message processing traceback:\n{traceback_str}")

        st.session_state.messages.append({"role": "error", "content": error_msg})

        # Show debug info in UI
        st.error(f"🔍 [DEBUG] Exception: {repr(e)}")
        with st.expander("🔍 Debug Details"):
            st.code(traceback_str)

    st.rerun()

# Footer
st.caption(
    "🚀 Được hỗ trợ bởi LangGraph & Gemini AI | Cuộc trò chuyện đa lượt với InMemorySaver"
)
