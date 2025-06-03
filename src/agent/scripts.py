"""
Conversation Scripts Handler
Manages predefined scripts for standard conversation scenarios
"""

import os
import pandas as pd
from typing import Dict, Optional


class ConversationScripts:
    """Handler for conversation scripts and standard responses"""

    def __init__(self, scripts_file: Optional[str] = None):
        self.scripts = {}
        self.scripts_file = scripts_file

        # Default built-in scripts (fallback)
        self._load_default_scripts()

        # Load from file if provided
        if scripts_file and os.path.exists(scripts_file):
            self._load_scripts_from_file(scripts_file)
            print(f"✅ Loaded conversation scripts from {scripts_file}")
        else:
            print("⚠️ Using default built-in scripts")

    def _load_default_scripts(self):
        """Load default conversation scripts"""
        self.scripts = {
            "greeting": """Xin chào! Em là Trợ lý ảo Viettelpay Pro sẵn sàng hỗ trợ Anh/chị!
Hiện tại, Trợ lý ảo đang trong giai đoạn thử nghiệm hỗ trợ nghiệp vụ cước viễn thông, thẻ cào và thẻ game với các nội dung sau:
 - Hướng dẫn sử dụng
 - Chính sách phí bán hàng
 - Tìm hiểu quy định hủy giao dịch
 - Hướng dẫn xử lý một số lỗi thường gặp.

Anh/Chị vui lòng bấm vào từng chủ đề để xem chi tiết.
Nếu thông tin chưa đáp ứng nhu cầu, Anh/Chị hãy đặt lại câu hỏi để em tiếp tục hỗ trợ ạ!""",
            "out_of_scope": """Cảm ơn Anh/chị đã đặt câu hỏi!
Trợ lý ảo Viettelpay Pro đang thử nghiệm và cập nhật kiến thức nghiệp vụ để hỗ trợ Anh/chị tốt hơn. Vì vậy, rất tiếc nhu cầu hiện tại của Anh/chị nằm ngoài khả năng hỗ trợ của Trợ lý ảo.

Để được hỗ trợ chính xác và đầy đủ hơn, Anh/chị vui lòng gửi yêu cầu hỗ trợ tại đây""",
            "human_request_attempt_1": """Anh/Chị vui lòng chia sẻ thêm nội dung cần hỗ trợ, Em rất mong được giải đáp trực tiếp để tiết kiệm thời gian của Anh/Chị ạ!""",
            "human_request_attempt_2": """Rất tiếc! Hiện tại hệ thống chưa có Tư vấn viên hỗ trợ trực tuyến.
Tuy nhiên, Anh/chị vẫn có thể yêu cầu hỗ trợ được trợ giúp qua các hình thức sau: 
📌 1. Đặt câu hỏi ngay tại đây, Trợ lý ảo ViettelPay Pro luôn sẵn sàng hỗ trợ Anh/chị trong phạm vi nghiệp vụ thử nghiệm (nghiệp vụ cước viễn thông, thẻ cào, thẻ game):
✅ Hướng dẫn sử dụng
✅ Chính sách phí bán hàng
✅ Tìm hiểu về quy định hủy giao dịch
✅ Hướng dẫn xử lý một số lỗi thường gặp.
📌 2. Tìm hiểu thông tin nghiệp vụ tại mục:
📚 Hướng dẫn, hỗ trợ: Các video hướng dẫn nghiệp vụ
💡Thông báo: Các tin tức nghiệp vụ và tin nâng cấp hệ thống/tin sự cố.
📌 3. Gửi yêu cầu hỗ trợ tại đây
 Hoặc gọi Tổng đài 1789 nhánh 5 trong trường hợp khẩn cấp""",
            "confirmation_check": """Anh/Chị có thắc mắc thêm vấn đề nào liên quan đến nội dung em vừa cung cấp không ạ?""",
            "closing": """Hy vọng những thông tin vừa rồi đã đáp ứng nhu cầu của Anh/chị.
Nếu cần hỗ trợ thêm, Anh/Chị hãy đặt câu hỏi để em tiếp tục hỗ trợ ạ!
🌟 Chúc Anh/chị một ngày thật vui và thành công!""",
            "ask_for_clarity": """Em chưa hiểu rõ yêu cầu của Anh/chị. Anh/chị có thể chia sẻ cụ thể hơn được không ạ?""",
            "empathy_error": """Em hiểu Anh/chị đang gặp khó khăn với lỗi này. Để hỗ trợ Anh/chị tốt nhất, em sẽ tìm hiểu và đưa ra hướng giải quyết cụ thể.""",
        }

    def _load_scripts_from_file(self, file_path: str):
        """Load scripts from CSV file (kich_ban.csv format)"""
        try:
            df = pd.read_csv(file_path)

            # Map CSV scenarios to script keys
            scenario_mapping = {
                "Chào hỏi": "greeting",
                "Trao đổi thông tin chính": "out_of_scope",  # First occurrence
                "Trước khi kết thúc phiên": "confirmation_check",
                "Kết thúc": "closing",
            }

            for _, row in df.iterrows():
                scenario_type = row.get("Loại kịch bản", "")
                situation = row.get("Tình huống", "")
                script = row.get("Kịch bản chốt", "")

                # Handle specific mappings
                if scenario_type == "Chào hỏi":
                    self.scripts["greeting"] = script
                elif scenario_type == "Trao đổi thông tin chính":
                    if "ngoài nghiệp vụ" in situation:
                        self.scripts["out_of_scope"] = script
                    elif "gặp tư vấn viên" in situation:
                        if "Lần 1" in script:
                            self.scripts["human_request_attempt_1"] = (
                                script.split("Lần 1:")[1].split("Lần 2:")[0].strip()
                            )
                        if "Lần 2" in script:
                            self.scripts["human_request_attempt_2"] = script.split(
                                "Lần 2:"
                            )[1].strip()
                    elif "không đủ ý" in situation:
                        self.scripts["ask_for_clarity"] = (
                            "Em chưa hiểu rõ yêu cầu của Anh/chị. Anh/chị có thể chia sẻ cụ thể hơn được không ạ?"
                        )
                    elif "lỗi" in situation:
                        self.scripts["empathy_error"] = (
                            "Em hiểu Anh/chị đang gặp khó khăn với lỗi này."
                        )
                elif scenario_type == "Trước khi kết thúc phiên":
                    self.scripts["confirmation_check"] = script
                elif scenario_type == "Kết thúc":
                    self.scripts["closing"] = script

        except Exception as e:
            print(f"⚠️ Error loading scripts from file: {e}")
            print("Using default scripts instead")

    def get_script(self, script_type: str) -> Optional[str]:
        """Get script by type"""
        return self.scripts.get(script_type)

    def has_script(self, script_type: str) -> bool:
        """Check if script exists"""
        return script_type in self.scripts

    def get_all_script_types(self) -> list:
        """Get all available script types"""
        return list(self.scripts.keys())

    def add_script(self, script_type: str, script_content: str):
        """Add or update a script"""
        self.scripts[script_type] = script_content

    def get_stats(self) -> dict:
        """Get statistics about loaded scripts"""
        return {
            "total_scripts": len(self.scripts),
            "script_types": list(self.scripts.keys()),
            "source": "file" if self.scripts_file else "default",
        }


# Usage example
if __name__ == "__main__":
    # Test with default scripts
    scripts = ConversationScripts()
    print("📊 Scripts Stats:", scripts.get_stats())

    # Test specific scripts
    greeting = scripts.get_script("greeting")
    print(f"\n👋 Greeting Script:\n{greeting}")

    # Test loading from file (if available)
    try:
        scripts_with_file = ConversationScripts(
            "./viettelpay_docs/processed/kich_ban.csv"
        )
        print(f"\n📊 File-based Scripts Stats: {scripts_with_file.get_stats()}")
    except Exception as e:
        print(f"File loading test failed: {e}")
