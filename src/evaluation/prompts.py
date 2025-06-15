"""
Prompts for ViettelPay Synthetic Evaluation Dataset Creation
Simplified version for MRR and Hit Rate evaluation only
"""

# Question Generation Prompt (JSON format for better parsing)
QUESTION_GENERATION_PROMPT = """Bạn là chuyên gia tạo câu hỏi đánh giá cho hệ thống ViettelPay Pro.
ViettelPay Pro là ứng dụng chuyên biệt dành cho các đại lý và điểm bán của Viettel, giúp họ thực hiện các giao dịch tài chính và viễn thông cho khách hàng một cách nhanh chóng, an toàn và đơn giản.
Dựa trên đoạn văn bản sau từ tài liệu hướng dẫn, hãy tạo ra {num_questions} câu hỏi đa dạng:

<context>
{content}
</context>

Tạo các loại câu hỏi:
1. Câu hỏi trực tiếp về thông tin trong đoạn văn
2. Câu hỏi về cách thực hiện hoặc quy trình  
3. Câu hỏi về lỗi, vấn đề hoặc troubleshooting
4. Câu hỏi về quy định, chính sách, phí

Yêu cầu cho mỗi câu hỏi:
- Tự nhiên như khách hàng ViettelPay thật sẽ hỏi
- Có thể trả lời được từ đoạn văn bản đã cho
- Ngắn gọn (5-20 từ)
- Sử dụng tiếng Việt thông dụng
- Đa dạng về loại câu hỏi và độ phức tạp

QUAN TRỌNG: Trả về kết quả dưới dạng JSON với format chính xác như sau:
{{
  "questions": [
    "Câu hỏi đầu tiên?",
    "Câu hỏi thứ hai?",
    "Câu hỏi thứ ba?"
  ]
}}

CHỈ trả về JSON, không có text khác."""

# Multi-Turn Conversation Generation Prompt (JSON format)
MULTI_TURN_CONVERSATION_GENERATION_PROMPT = """Bạn là một chuyên gia trong việc tạo dữ liệu huấn luyện cho chatbot hỗ trợ khách hàng của ứng dụng ViettelPay Pro.
ViettelPay Pro là ứng dụng chuyên biệt dành cho các đại lý và điểm bán của Viettel, giúp họ thực hiện các giao dịch tài chính và viễn thông cho khách hàng một cách nhanh chóng, an toàn và đơn giản.
Nhiệm vụ của bạn là tạo ra **{num_conversations}** cuộc hội thoại đa lượt, chân thực và tự nhiên giữa người dùng (Đại lý/Điểm bán) và Trợ lý AI. Toàn bộ nội dung cuộc hội thoại phải dựa **hoàn toàn** vào thông tin được cung cấp trong tài liệu dưới đây.

<context>
{content}
</context>

Hãy tạo ra các cuộc hội thoại xoay quanh những kịch bản sau:

1.  **Giải quyết vấn đề (`error_resolution`):** Người dùng gặp lỗi, giao dịch thất bại, hoặc một tính năng không hoạt động như mong đợi. Họ muốn tìm hiểu nguyên nhân và cách khắc phục.
    * *Ví dụ luồng hội thoại:* Báo lỗi -> Hỏi về nguyên nhân sâu xa -> Hỏi cách để tránh lỗi này trong tương lai.

2.  **Hướng dẫn thực hiện (`procedure_guide`):** Người dùng muốn biết cách thực hiện một tác vụ cụ thể. Cuộc hội thoại nên đi sâu vào các bước, điều kiện, hoặc các chi tiết liên quan.
    * *Ví dụ luồng hội thoại:* Hỏi cách thực hiện một dịch vụ -> Hỏi về một bước cụ thể -> Hỏi về một trường hợp đặc biệt (ví dụ: "nếu làm cho nhà mạng khác thì sao?").

3.  **Tra cứu thông tin (`policy_info`):** Người dùng có câu hỏi về chính sách, phí, hạn mức, hoặc các quy định của dịch vụ.
    * *Ví dụ luồng hội thoại:* Hỏi về một quy định chung -> Hỏi về một trường hợp áp dụng cụ thể -> Hỏi về các ngoại lệ.

**YÊU CẦU QUAN TRỌNG:**

* **Dòng chảy tự nhiên:** Mỗi lượt hỏi của người dùng phải là một phản ứng logic, tự nhiên sau khi nhận được câu trả lời (tưởng tượng) từ AI. Hãy hình dung AI đã đưa ra câu trả lời hữu ích nhưng chưa đầy đủ, khiến người dùng phải hỏi thêm để làm rõ.
* **Chân thực như người dùng thật:**
    * Sử dụng ngôn ngữ đời thường, ngắn gọn, đi thẳng vào vấn đề.
    * Có thể dùng các từ viết tắt phổ biến (vd: "sđt", "tk", "gd", "đk").
    * Giọng điệu có thể thể hiện sự bối rối, cần hỗ trợ gấp hoặc tò mò.
* **Bám sát tài liệu:** **Không** được tự ý sáng tạo thông tin, chính sách, hoặc tính năng không có trong phần `<context>`.
* Tất cả các lượt câu hỏi, đặc biệt là câu hỏi cuối cùng, phải có thể trả lời được từ thông tin trong tài liệu `<context>`.
* **Cấu trúc:** Mỗi cuộc hội thoại phải có từ 2 đến 3 lượt hỏi từ phía người dùng.
* **Ngôn ngữ:** Tiếng Việt.

Ví dụ cuộc hội thoại:
```
Lượt 1: "mã lỗi 606 là gì vậy?"
Lượt 2: "làm sao để khắc phục lỗi này?"
```

QUAN TRỌNG: Trả về kết quả dưới dạng JSON với format chính xác như sau:
{{
  "conversations": [
    {{
      "type": "error_resolution",
      "turns": [
        {{"role": "user", "content": "mã lỗi 606 là gì vậy?"}},
        {{"role": "user", "content": "làm sao để khắc phục lỗi này?"}}
      ]
    }},
    {{
      "type": "procedure_inquiry", 
      "turns": [
        {{"role": "user", "content": "nạp cước điện thoại như thế nào?"}},
        {{"role": "user", "content": "có thể nạp cho số Viettel không?"}},
        {{"role": "user", "content": "có các mệnh giá nào?"}}
      ]
    }}
  ]
}}

CHỈ trả về JSON, không có text khác."""


# Quality Check Prompt for Generated Questions
QUESTION_QUALITY_CHECK_PROMPT = """Đánh giá chất lượng của câu hỏi được tạo ra cho hệ thống ViettelPay Pro. ViettelPay Pro là ứng dụng chuyên biệt dành cho các đại lý và điểm bán của Viettel, giúp họ thực hiện các giao dịch tài chính và viễn thông cho khách hàng một cách nhanh chóng, an toàn và đơn giản.

Câu hỏi: {question}
Đoạn văn gốc: {context}

Tiêu chí đánh giá:
1. Clarity (Rõ ràng): Câu hỏi có dễ hiểu không?
2. Answerability (Có thể trả lời): Có thể trả lời từ đoạn văn không?
3. Naturalness (Tự nhiên): Có giống cách khách hàng thật hỏi không?
4. Relevance (Liên quan): Có phù hợp với nội dung ViettelPay không?

Mỗi tiêu chí từ 1-5 điểm (5 là tốt nhất).

QUAN TRỌNG: Trả về kết quả dưới dạng JSON với format chính xác như sau:
{{
  "clarity": 5,
  "answerability": 4, 
  "naturalness": 5,
  "relevance": 5,
  "overall_score": 4.75,
  "keep_question": true,
  "feedback": "Câu hỏi tốt, rõ ràng và tự nhiên"
}}

CHỈ trả về JSON, không có text khác."""

# Context Quality Check Prompt
CONTEXT_QUALITY_CHECK_PROMPT = """Đánh giá chất lượng của đoạn văn bản ViettelPay Pro để tạo câu hỏi. ViettelPay Pro là ứng dụng chuyên biệt dành cho các đại lý và điểm bán của Viettel, giúp họ thực hiện các giao dịch tài chính và viễn thông cho khách hàng một cách nhanh chóng, an toàn và đơn giản.

Đoạn văn bản:
{content}

Tiêu chí đánh giá:
1. Clarity (Rõ ràng): Thông tin có dễ hiểu không?
2. Completeness (Đầy đủ): Có đủ thông tin để tạo câu hỏi không?
3. Structure (Cấu trúc): Có tổ chức tốt không?
4. Relevance (Liên quan): Có phù hợp với ViettelPay không?
5. Information_density (Mật độ thông tin): Có đủ thông tin hữu ích không?

Mỗi tiêu chí từ 1-5 điểm (5 là tốt nhất).

QUAN TRỌNG: Trả về kết quả dưới dạng JSON với format chính xác như sau:
{{
  "clarity": 5,
  "completeness": 4,
  "structure": 4,
  "relevance": 5,
  "information_density": 4,
  "overall_score": 4.4,
  "use_context": true,
  "feedback": "Đoạn văn tốt, có thể tạo câu hỏi chất lượng"
}}

CHỈ trả về JSON, không có text khác."""

# Question Evolution/Variation Prompt
QUESTION_EVOLUTION_PROMPT = """Tạo các biến thể của câu hỏi ViettelPay Pro để tăng tính đa dạng. ViettelPay Pro là ứng dụng chuyên biệt dành cho các đại lý và điểm bán của Viettel, giúp họ thực hiện các giao dịch tài chính và viễn thông cho khách hàng một cách nhanh chóng, an toàn và đơn giản.

Câu hỏi gốc: {original_question}
Ngữ cảnh: {context}

Tạo 3 biến thể khác nhau:
1. Phiên bản casual/thông tục (cách nói hàng ngày)
2. Phiên bản formal/lịch sự (cách nói trang trọng)  
3. Phiên bản cụ thể (thêm chi tiết, tình huống cụ thể)

Yêu cầu:
- Giữ nguyên ý nghĩa cốt lõi
- Vẫn có thể trả lời từ cùng ngữ cảnh
- Tự nhiên với người dùng Việt Nam
- Đa dạng về cách diễn đạt

QUAN TRỌNG: Trả về kết quả dưới dạng JSON với format chính xác như sau:
{{
  "original_question": "{original_question}",
  "variations": [
    "Phiên bản casual",
    "Phiên bản formal", 
    "Phiên bản cụ thể"
  ]
}}

CHỈ trả về JSON, không có text khác."""

# Dataset Statistics Prompt
DATASET_STATS_PROMPT = """Phân tích thống kê dataset đánh giá ViettelPay Pro. ViettelPay Pro là ứng dụng chuyên biệt dành cho các đại lý và điểm bán của Viettel, giúp họ thực hiện các giao dịch tài chính và viễn thông cho khách hàng một cách nhanh chóng, an toàn và đơn giản.

Dữ liệu:
- Tổng số câu hỏi: {total_questions}
- Tổng số documents: {total_documents}  
- Câu hỏi theo loại: {question_types}

Tạo báo cáo thống kê và đề xuất cải thiện.

QUAN TRỌNG: Trả về kết quả dưới dạng JSON với format chính xác như sau:
{{
  "coverage_analysis": {{
    "error_handling": "20%",
    "procedures": "30%", 
    "policies": "25%",
    "faq": "25%"
  }},
  "quality_metrics": {{
    "avg_questions_per_doc": 2.1,
    "question_diversity": "high"
  }},
  "recommendations": [
    "Tăng câu hỏi về error handling",
    "Cân bằng độ khó của câu hỏi"
  ]
}}

CHỈ trả về JSON, không có text khác."""

# Error Analysis Prompt
ERROR_ANALYSIS_PROMPT = """Phân tích lỗi trong quá trình đánh giá retrieval ViettelPay Pro. ViettelPay Pro là ứng dụng chuyên biệt dành cho các đại lý và điểm bán của Viettel, giúp họ thực hiện các giao dịch tài chính và viễn thông cho khách hàng một cách nhanh chóng, an toàn và đơn giản.

Kết quả đánh giá:
{evaluation_results}

Xác định:
1. Câu hỏi có hiệu suất thấp (Hit Rate < 0.3)
2. Loại lỗi thường gặp
3. Nguyên nhân gốc rễ
4. Đề xuất cải thiện

QUAN TRỌNG: Trả về kết quả dưới dạng JSON với format chính xác như sau:
{{
  "low_performance_queries": [
    {{"query": "câu hỏi", "hit_rate": 0.2, "issue": "từ khóa không rõ ràng"}}
  ],
  "common_error_types": [
    "Thiếu từ khóa chính",
    "Ngữ cảnh không đủ",
    "Chunking không tối ưu"
  ],
  "improvement_suggestions": [
    "Cải thiện chunking strategy",
    "Thêm synonyms cho từ khóa"
  ]
}}

CHỈ trả về JSON, không có text khác."""

# Intent Classification Prompt
# Updated Intent Classification Conversation Generation Prompt with chunk mixing support
# Improved Intent Classification Conversation Generation Prompt
INTENT_CLASSIFICATION_CONVERSATION_GENERATION_PROMPT = """Bạn là chuyên gia tạo dữ liệu đánh giá cho hệ thống phân loại ý định (intent classification) của Trợ lý AI trên ứng dụng ViettelPay Pro.
ViettelPay Pro là ứng dụng chuyên biệt dành cho các đại lý và điểm bán của Viettel, giúp họ thực hiện các giao dịch tài chính và viễn thông cho khách hàng một cách nhanh chóng, an toàn và đơn giản.

Nhiệm vụ của bạn là tạo ra **{num_conversations}** cuộc hội thoại đa lượt thực tế. Mỗi tin nhắn của người dùng phải được gán một nhãn `intent` chính xác.
`intent` là ý định của người dùng trong câu hỏi hiện tại và liên quan đến các lượt hỏi trước đó trong cuộc hội thoại.

**1. Định nghĩa các loại ý định (Bắt buộc phải tuân theo):**

* **`greeting`**: Chỉ là lời chào hỏi thuần túy, không có câu hỏi hoặc yêu cầu cụ thể nào khác. Nếu tin nhắn có cả lời chào VÀ câu hỏi thì phân loại theo các ý định khác, không phải greeting.
    * *Ví dụ:* "chào em", "hello shop", "xin chào ạ"
    * *Không phải greeting:* "xin chào, cho hỏi về lỗi 606" → đây là error_help
* **`faq`**: Các câu hỏi đáp chung, tìm hiểu về dịch vụ, tính năng, v.v.
    * *Ví dụ:* "App có bán thẻ game không?", "ViettelPay Pro nạp tiền được cho mạng nao?"
* **`error_help`**: Báo cáo sự cố, hỏi về mã lỗi cụ thể.
    * *Ví dụ:* "Giao dịch báo lỗi 606", "tại sao tôi không thanh toán được?", "lỗi này là gì?"
* **`procedure_guide`**: Hỏi về các bước cụ thể để thực hiện một tác vụ.
    * *Ví dụ:* "làm thế nào để hủy giao dịch?", "chỉ tôi cách lấy lại mã thẻ cào", "hướng dẫn nạp cước"
* **`human_request`**: Yêu cầu được nói chuyện trực tiếp với nhân viên hỗ trợ.
    * *Ví dụ:* "cho tôi gặp người thật", "nối máy cho tổng đài", "em k hiểu, cho gặp ai đó"
* **`out_of_scope`**: Câu hỏi ngoài phạm vi ViettelPay (thời tiết, chính trị, v.v.), không liên quan gì đến các dịch vụ tài chính, viễn thông của Viettel.
    * *Ví dụ:* "dự báo thời tiết hôm nay?", "giá xăng bao nhiêu?", "cách nấu phở"
* **`unclear`**: Câu hỏi không rõ ràng, thiếu thông tin cụ thể, cần người dùng bổ sung thêm chi tiết để có thể hỗ trợ hiệu quả.
    * *Ví dụ:* "lỗi", "giúp với", "gd", "???", "ko hiểu", "bị lỗi giờ sao đây", "không thực hiện được", "sao vậy", "tại sao thế"

**2. Nguồn kiến thức (Context):**

Sử dụng tài liệu dưới đây để lấy các mã lỗi (vd: 606, W02), tên dịch vụ (vd: Gạch nợ cước), và các tình huống thực tế để xây dựng cuộc hội thoại.

<context>
{content}
</context>

**3. Yêu cầu về kịch bản hội thoại:**

* Mỗi cuộc hội thoại có từ 2 đến 4 lượt hỏi từ người dùng. Tùy chỉnh sao cho phù hợp với tài liệu.
* {generation_instruction}
* Tạo các cuộc hội thoại đa dạng, không lặp lại.
* **QUAN TRỌNG - Intent Mixing**: Khoảng 70% tin nhắn nên liên quan đến context, nhưng 30% tin nhắn nên là các intent tự nhiên khác như:
  - `greeting` ở đầu cuộc hội thoại
  - `unclear` khi người dùng hỏi không rõ ràng, trợ lý cần thêm thông tin từ người dùng
  - `human_request` khi họ muốn hỗ trợ trực tiếp
  - `out_of_scope` khi họ hỏi không liên quan đến ViettelPay Pro
* Ngôn ngữ phải tự nhiên như người dùng thật, có thể dùng từ viết tắt, và đi thẳng vào vấn đề.
* Tạo các tình huống thực tế như người dùng thật sự sẽ hỏi. Không cần phải kết thúc bằng cảm ơn hay tạm biệt.
* Các câu hỏi nên dễ để phân loại ý định, không cần phải suy nghĩ quá lâu.

**4. Ví dụ cuộc hội thoại thực tế:**
{{
  "conversations": [
    {{
      "turns": [
        {{"user": "chào em", "intent": "greeting"}},
        {{"user": "mã lỗi 606 là gì z?", "intent": "error_help"}},
        {{"user": "em ko hiểu, gặp ai đó dc ko?", "intent": "human_request"}}
      ]
    }},
    {{
      "turns": [
        {{"user": "nạp cước như nào?", "intent": "procedure_guide"}},
        {{"user": "hôm nay trời đẹp nhỉ", "intent": "out_of_scope"}},
        {{"user": "ờ quay lại, có phí ko?", "intent": "faq"}}
      ]
    }}
  ]
}}

**5. Định dạng đầu ra (Output):**

QUAN TRỌNG: Trả về kết quả dưới dạng JSON với format chính xác như ví dụ trên.
CHỈ trả về JSON, không có text khác."""
