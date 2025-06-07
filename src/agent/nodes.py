"""
LangGraph Agent State and Processing Nodes
"""

from typing import Dict, List, Optional, TypedDict, Annotated
from langchain.schema import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
import json
import re


class ViettelPayState(TypedDict):
    """State for ViettelPay agent workflow with message history support"""

    # Message history for multi-turn conversation
    messages: Annotated[List[AnyMessage], add_messages]

    # Processing
    intent: Optional[str]
    confidence: Optional[float]

    # Query enhancement
    enhanced_query: Optional[str]

    # Knowledge retrieval
    retrieved_docs: Optional[List[Document]]

    # Response type metadata
    response_type: Optional[str]  # "script" or "generated"

    # Metadata
    error: Optional[str]
    processing_info: Optional[Dict]


def get_conversation_context(messages: List[AnyMessage], max_messages: int = 6) -> str:
    """
    Extract conversation context from message history

    Args:
        messages: List of conversation messages
        max_messages: Maximum number of recent messages to include

    Returns:
        Formatted conversation context string
    """
    if len(messages) <= 1:
        return ""

    context = "\n\nLịch sử cuộc hội thoại:\n"
    # Get recent messages (excluding the current/last message for intent classification)
    recent_messages = messages[
        -(max_messages + 1) : -1
    ]  # Exclude the very last message

    for msg in recent_messages:
        # Handle different message types more robustly
        if hasattr(msg, "type"):
            if msg.type == "human":
                role = "Người dùng"
            elif msg.type == "ai":
                role = "Trợ lý"
            else:
                role = f"Unknown-{msg.type}"
        elif hasattr(msg, "role"):
            if msg.role in ["user", "human"]:
                role = "Người dùng"
            elif msg.role in ["assistant", "ai"]:
                role = "Trợ lý"
            else:
                role = f"Unknown-{msg.role}"
        else:
            role = "Unknown"

        # Limit message length to avoid token overflow
        # content = msg.content[:1000] + "..." if len(msg.content) > 1000 else msg.content
        content = msg.content
        context += f"{role}: {content}\n"
    # print(context)
    return context


def classify_intent_node(state: ViettelPayState, llm_client) -> ViettelPayState:
    """Node for intent classification using LLM with conversation context"""

    # Get the latest user message
    messages = state["messages"]
    if not messages:
        return {
            **state,
            "intent": "unclear",
            "confidence": 0.0,
            "error": "No messages found",
        }

    # Find the last human/user message
    user_message = None
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            user_message = msg.content
            break
        elif hasattr(msg, "role") and msg.role == "user":
            user_message = msg.content
            break

    if not user_message:
        return {
            **state,
            "intent": "unclear",
            "confidence": 0.0,
            "error": "No user message found",
        }

    try:
        # Get conversation context for better intent classification
        conversation_context = get_conversation_context(messages)

        # Intent classification prompt with context
        classification_prompt = f"""
Bạn là hệ thống phân loại ý định cho ViettelPay Pro. Phân tích tin nhắn của người dùng và trả về ý định chính.

Các loại ý định:
- greeting: Chào hỏi, bắt đầu cuộc trò chuyện
- faq: Câu hỏi về dịch vụ, tính năng, thông tin ViettelPay
- error_help: Báo lỗi, cần hỗ trợ xử lý lỗi, mã lỗi
- procedure_guide: Hỏi cách thực hiện một thao tác, quy trình
- policy_info: Hỏi về quy định, chính sách
- human_request: Yêu cầu nói chuyện với người thật, tư vấn viên
- out_of_scope: Câu hỏi ngoài phạm vi ViettelPay (thời tiết, chính trị, v.v.)
- unclear: Câu hỏi không rõ ràng, không hiểu

Ví dụ:
- "mã lỗi 606 là gì" → error_help
- "làm sao nạp cước" → procedure_guide  
- "ViettelPay hỗ trợ mạng nào" → faq
- "làm sao khắc phục?" (sau khi hỏi về lỗi) → error_help
- "tôi vẫn chưa hiểu" (sau câu trả lời) → yêu cầu giải thích thêm, phụ thuộc vào ngữ cảnh

{conversation_context}

Tin nhắn hiện tại: "{user_message}"

Hãy phân tích dựa trên cả ngữ cảnh cuộc hội thoại và tin nhắn hiện tại.

QUAN TRỌNG: Chỉ trả về JSON thuần túy, không có text khác. Format chính xác:
{{"intent": "tên_ý_định", "confidence": 0.9, "explanation": "lý do ngắn gọn"}}
"""

        # Get classification using the pre-initialized LLM client
        response = llm_client.generate(classification_prompt, temperature=0.1)

        print(f"🔍 Raw LLM response: {response}")

        # Parse JSON response
        try:
            # Try to extract JSON from response (in case there's extra text)
            response_clean = response.strip()

            # Look for JSON object in the response
            json_match = re.search(r"\{.*\}", response_clean, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
            else:
                # Try parsing the whole response
                result = json.loads(response_clean)

            intent = result.get("intent", "unclear")
            confidence = result.get("confidence", 0.5)
            explanation = result.get("explanation", "")

            print(
                f"✅ JSON parsed successfully: intent={intent}, confidence={confidence}"
            )

        except (json.JSONDecodeError, AttributeError) as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"   Raw response: {response}")

            # Fallback: try to extract intent from text
            response_lower = response.lower()
            if any(
                word in response_lower for word in ["lỗi", "error", "606", "mã lỗi"]
            ):
                intent = "error_help"
                confidence = 0.7
            elif any(word in response_lower for word in ["xin chào", "hello", "chào"]):
                intent = "greeting"
                confidence = 0.8
            elif any(word in response_lower for word in ["hủy", "cancel", "thủ tục"]):
                intent = "procedure_guide"
                confidence = 0.7
            elif any(
                word in response_lower for word in ["nạp", "cước", "dịch vụ", "faq"]
            ):
                intent = "faq"
                confidence = 0.7
            else:
                intent = "unclear"
                confidence = 0.3

            print(f"🔄 Fallback classification: {intent} (confidence: {confidence})")
            explanation = "Fallback classification due to JSON parse error"

        print(f"🎯 Intent classified: {intent} (confidence: {confidence})")

        return {
            **state,
            "intent": intent,
            "confidence": confidence,
            "processing_info": {
                "classification_raw": response,
                "explanation": explanation,
                "context_used": bool(conversation_context.strip()),
            },
        }

    except Exception as e:
        print(f"❌ Intent classification error: {e}")
        return {**state, "intent": "unclear", "confidence": 0.0, "error": str(e)}


def query_enhancement_node(state: ViettelPayState, llm_client) -> ViettelPayState:
    """Node for enhancing search query using conversation context"""

    # Get the latest user message
    messages = state["messages"]
    if not messages:
        return {**state, "enhanced_query": "", "error": "No messages found"}

    # Find the last human/user message
    user_message = None
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            user_message = msg.content
            break
        elif hasattr(msg, "role") and msg.role == "user":
            user_message = msg.content
            break

    if not user_message:
        return {**state, "enhanced_query": "", "error": "No user message found"}

    try:
        # Get conversation context
        conversation_context = get_conversation_context(messages)

        # If no context, use original message
        if not conversation_context.strip():
            print(f"🔍 No context available, using original query: {user_message}")
            return {**state, "enhanced_query": user_message}

        # Query enhancement prompt
        enhancement_prompt = f"""
Bạn là hệ thống tạo truy vấn tìm kiếm thông minh cho ViettelPay Pro. Hãy tạo truy vấn tìm kiếm tối ưu dựa trên toàn bộ ngữ cảnh cuộc trò chuyện.

{conversation_context}

Tin nhắn hiện tại: "{user_message}"

Nhiệm vụ: Tạo truy vấn tìm kiếm chi tiết và thông minh bằng cách:

1. **Phân tích ngữ cảnh tổng thể:**
   - Kết hợp thông tin từ bối cảnh cuộc trò chuyện trước
   - Hiểu mối liên hệ giữa tin nhắn hiện tại và chủ đề đã thảo luận
   - Nhận diện chuỗi vấn đề hoặc yêu cầu liên quan

2. **Mở rộng và làm rõ truy vấn:**
   - Thay thế đại từ ("nó", "thế", "vậy", "đó") hoặc các từ không rõ ràng bằng đối tượng cụ thể từ ngữ cảnh
   - Thêm từ khóa liên quan và thuật ngữ chuyên môn ViettelPay
   - Bao gồm các biến thể cách diễn đạt và từ đồng nghĩa
   - Cụ thể hóa các yêu cầu mơ hồ dựa trên ngữ cảnh

3. **Tối ưu cho tìm kiếm:**
   - Sử dụng thuật ngữ chính xác của ViettelPay (giao dịch, nạp cước, OTP, mã lỗi...)
   - Sử dụng từ đồng nghĩa và cách diễn đạt khác nhau trong tiếng Việt để tăng khả năng tìm kiếm
   - Thêm các từ khóa và cụm từ có khả năng xuất hiện trong tài liệu hướng dẫn vào một câu riêng cuối truy vấn
   - Duy trì tính tự nhiên của tiếng Việt
   - Ưu tiên độ chính xác và liên quan

QUAN TRỌNG: 
- Truy vấn phải phản ánh đúng ý định và ngữ cảnh đầy đủ
- Không thêm thông tin không có trong ngữ cảnh
- Duy trì tính tự nhiên của tiếng Việt
- Tập trung vào việc tìm kiếm thông tin hỗ trợ cụ thể

CHỈ trả về truy vấn được tăng cường, không có giải thích.
"""

        # Get enhanced query
        enhanced_query = llm_client.generate(enhancement_prompt, temperature=0.1)
        enhanced_query = enhanced_query.strip()

        print(f"🔍 Original query: {user_message}")
        print(f"🚀 Enhanced query: {enhanced_query}")

        return {**state, "enhanced_query": enhanced_query}

    except Exception as e:
        print(f"❌ Query enhancement error: {e}")
        # Fallback to original message
        return {**state, "enhanced_query": user_message, "error": str(e)}


def knowledge_retrieval_node(
    state: ViettelPayState, knowledge_retriever
) -> ViettelPayState:
    """Node for knowledge retrieval using pre-initialized ViettelKnowledgeBase"""

    # Use enhanced query if available, otherwise fall back to extracting from messages
    enhanced_query = state.get("enhanced_query", "")

    if not enhanced_query:
        # Fallback: extract from messages
        messages = state["messages"]
        if not messages:
            return {**state, "retrieved_docs": [], "error": "No messages found"}

        # Find the last human/user message
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                enhanced_query = msg.content
                break
            elif hasattr(msg, "role") and msg.role == "user":
                enhanced_query = msg.content
                break

    if not enhanced_query:
        return {**state, "retrieved_docs": [], "error": "No query available"}

    try:
        if not knowledge_retriever:
            raise ValueError("Knowledge retriever not available")

        # Retrieve relevant documents using enhanced query and pre-initialized ViettelKnowledgeBase
        retrieved_docs = knowledge_retriever.search(enhanced_query, top_k=10)

        print(
            f"📚 Retrieved {len(retrieved_docs)} documents for enhanced query: {enhanced_query}"
        )

        return {**state, "retrieved_docs": retrieved_docs}

    except Exception as e:
        print(f"❌ Knowledge retrieval error: {e}")
        return {**state, "retrieved_docs": [], "error": str(e)}


def script_response_node(state: ViettelPayState) -> ViettelPayState:
    """Node for script-based responses"""

    from src.agent.scripts import ConversationScripts
    from langchain_core.messages import AIMessage

    intent = state.get("intent", "")

    try:
        # Load scripts
        scripts = ConversationScripts("./viettelpay_docs/processed/kich_ban.csv")

        # Map intents to script types
        intent_to_script = {
            "greeting": "greeting",
            "out_of_scope": "out_of_scope",
            "human_request": "human_request_attempt_1",  # Could be enhanced later
            "unclear": "ask_for_clarity",
        }

        script_type = intent_to_script.get(intent)

        if script_type and scripts.has_script(script_type):
            response_text = scripts.get_script(script_type)
            print(f"📋 Using script response: {script_type}")

            # Add AI message to the conversation
            ai_message = AIMessage(content=response_text)

            return {**state, "messages": [ai_message], "response_type": "script"}

        else:
            # Fallback script
            fallback_response = (
                "Xin lỗi, em chưa hiểu rõ yêu cầu của anh/chị. Vui lòng thử lại."
            )
            ai_message = AIMessage(content=fallback_response)

            print(f"📋 Using fallback script for intent: {intent}")

            return {**state, "messages": [ai_message], "response_type": "script"}

    except Exception as e:
        print(f"❌ Script response error: {e}")
        fallback_response = "Xin lỗi, em gặp lỗi kỹ thuật. Vui lòng thử lại sau."
        ai_message = AIMessage(content=fallback_response)

        return {
            **state,
            "messages": [ai_message],
            "response_type": "error",
            "error": str(e),
        }


def generate_response_node(state: ViettelPayState, llm_client) -> ViettelPayState:
    """Node for LLM-based response generation with conversation context"""

    from langchain_core.messages import AIMessage

    # Get the latest user message and conversation history
    messages = state["messages"]
    if not messages:
        ai_message = AIMessage(content="Xin lỗi, em không thể xử lý yêu cầu này.")
        return {**state, "messages": [ai_message], "response_type": "error"}

    # Find the last human/user message
    user_message = None
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            user_message = msg.content
            break
        elif hasattr(msg, "role") and msg.role == "user":
            user_message = msg.content
            break

    if not user_message:
        ai_message = AIMessage(content="Xin lỗi, em không thể xử lý yêu cầu này.")
        return {**state, "messages": [ai_message], "response_type": "error"}

    intent = state.get("intent", "")
    retrieved_docs = state.get("retrieved_docs", [])
    enhanced_query = state.get("enhanced_query", "")

    try:
        # Build context from retrieved documents
        context = ""
        if retrieved_docs:
            context = "\n\n".join(
                [
                    f"[{doc.metadata.get('doc_type', 'unknown')}] {doc.page_content}"
                    for doc in retrieved_docs
                ]
            )

        # Get conversation context using the helper function
        conversation_context = get_conversation_context(messages, max_messages=6)

        # Generate response prompt based on intent
        if intent == "error_help":
            system_prompt = """Bạn là chuyên gia hỗ trợ kỹ thuật ViettelPay Pro.
Thể hiện sự cảm thông với khó khăn của người dùng.
Cung cấp giải pháp cụ thể, từng bước.
Nếu cần hỗ trợ thêm, hướng dẫn liên hệ tổng đài.
Nếu có lịch sử cuộc hội thoại, hãy tham khảo để đưa ra câu trả lời phù hợp và có tính liên kết."""

        elif intent == "procedure_guide":
            system_prompt = """Bạn là hướng dẫn viên ViettelPay Pro.
Cung cấp hướng dẫn từng bước rõ ràng.
Bao gồm link video nếu có trong thông tin.
Sử dụng format có số thứ tự cho các bước.
Nếu có lịch sử cuộc hội thoại, hãy tham khảo để đưa ra câu trả lời phù hợp và có tính liên kết."""

        else:  # faq, policy_info, etc.
            system_prompt = """Bạn là trợ lý ảo ViettelPay Pro.
Trả lời câu hỏi dựa trên thông tin được cung cấp.
Giọng điệu thân thiện, chuyên nghiệp.
Sử dụng "Anh/chị" khi xưng hô.
Nếu có lịch sử cuộc hội thoại, hãy tham khảo để đưa ra câu trả lời phù hợp và có tính liên kết."""

        # Build full prompt with both knowledge context and conversation context
        generation_prompt = f"""{system_prompt}

Thông tin tham khảo từ cơ sở tri thức:
{context}

{conversation_context}

Câu hỏi hiện tại của người dùng: {user_message}
Truy vấn tìm kiếm đã được cải thiện: {enhanced_query}

Hãy trả lời câu hỏi dựa trên thông tin tham khảo và lịch sử cuộc hội thoại (nếu có). Nếu không có thông tin phù hợp, hãy nói rằng bạn cần thêm thông tin.
"""

        # Generate response using the pre-initialized LLM client
        response_text = llm_client.generate(generation_prompt, temperature=0.1)

        print(f"🤖 Generated response for intent: {intent}")

        # Add AI message to the conversation
        ai_message = AIMessage(content=response_text)

        return {**state, "messages": [ai_message], "response_type": "generated"}

    except Exception as e:
        print(f"❌ Response generation error: {e}")
        error_response = "Xin lỗi, em gặp lỗi khi xử lý yêu cầu. Vui lòng thử lại sau."
        ai_message = AIMessage(content=error_response)

        return {
            **state,
            "messages": [ai_message],
            "response_type": "error",
            "error": str(e),
        }


# Routing function for conditional edges
def route_after_intent_classification(state: ViettelPayState) -> str:
    """Route to appropriate node after intent classification"""

    intent = state.get("intent", "unclear")

    # Script-based intents (no knowledge retrieval needed)
    script_intents = {"greeting", "out_of_scope", "human_request", "unclear"}

    if intent in script_intents:
        return "script_response"
    else:
        # Knowledge-based intents need query enhancement first
        return "query_enhancement"


def route_after_query_enhancement(state: ViettelPayState) -> str:
    """Route after query enhancement (always to knowledge retrieval)"""
    return "knowledge_retrieval"


def route_after_knowledge_retrieval(state: ViettelPayState) -> str:
    """Route after knowledge retrieval (always to generation)"""
    return "generate_response"
