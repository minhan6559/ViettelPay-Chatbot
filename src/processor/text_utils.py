import re
from typing import List, Set

try:
    from underthesea import word_tokenize, pos_tag

    UNDERTHESEA_AVAILABLE = True
except ImportError:
    UNDERTHESEA_AVAILABLE = False
    print("[WARNING] underthesea not available, falling back to basic tokenization")


class VietnameseTextProcessor:
    """Vietnamese text processing utilities for ViettelPay knowledge base"""

    def __init__(self):
        # Vietnamese stop words
        self.vietnamese_stop_words = self._load_vietnamese_stop_words()

        # Keep important domain terms even if they appear in stop words
        self.domain_important_terms = {
            "lỗi",
            "error",
            "mã",
            "code",
            "bước",
            "step",
            "hướng",
            "dẫn",
            "guide",
            "thanh",
            "toán",
            "payment",
            "nạp",
            "cước",
            "topup",
            "mua",
            "buy",
            "viettel",
            "viettelpay",
            "app",
            "ứng",
            "dụng",
            "mobile",
            "thẻ",
            "card",
            "tiền",
            "money",
            "rút",
            "withdraw",
            "chuyển",
            "transfer",
        }

    def _load_vietnamese_stop_words(self) -> Set[str]:
        """Load Vietnamese stop words"""
        # Common Vietnamese stop words
        stop_words = {
            "và",
            "của",
            "có",
            "là",
            "được",
            "các",
            "một",
            "này",
            "cho",
            "với",
            "trong",
            "từ",
            "tại",
            "về",
            "như",
            "sau",
            "trước",
            "khi",
            "nếu",
            "để",
            "đã",
            "sẽ",
            "đang",
            "bị",
            "bởi",
            "theo",
            "những",
            "nhưng",
            "mà",
            "thì",
            "cũng",
            "hay",
            "hoặc",
            "nên",
            "phải",
            "rất",
            "lại",
            "chỉ",
            "đó",
            "đây",
            "kia",
            "nào",
            "ai",
            "gì",
            "sao",
            "đâu",
            "bao",
            "nhiều",
            "lắm",
            "hơn",
            "nhất",
            "cả",
            "tất",
            "mọi",
            "toàn",
            "chưa",
            "không",
            "chẳng",
            "đang",
            "vẫn",
            "còn",
            "đều",
            "cùng",
            "nhau",
            "riêng",
            "luôn",
            "ngay",
            "liền",
            "thêm",
            "nữa",
            "lần",
            "cuối",
            "đầu",
            "giữa",
            "ngoài",
            "trong",
            "trên",
            "dưới",
            "bên",
            "cạnh",
            "giữa",
            "trước",
            "sau",
            "gần",
            "xa",
            "cao",
            "thấp",
        }

        # Add English stop words that might appear
        english_stops = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
        }

        return stop_words.union(english_stops)

    def vietnamese_tokenize(self, text: str) -> List[str]:
        """Vietnamese word tokenization using underthesea or fallback"""
        if not text:
            return []

        if UNDERTHESEA_AVAILABLE:
            try:
                # Use underthesea for proper Vietnamese tokenization
                tokenized_text = word_tokenize(text, format="text")

                return tokenized_text.split()
            except Exception as e:
                print(
                    f"[WARNING] underthesea tokenization failed: {e}, falling back to basic"
                )

        # Fallback: basic tokenization with Vietnamese-aware splitting
        # Handle Vietnamese compound words better
        tokens = text.split()
        return [token.strip() for token in tokens if token.strip()]

    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """Remove Vietnamese stop words while preserving domain terms"""
        filtered_tokens = []

        for token in tokens:
            # Always keep domain-important terms
            if token.lower() in self.domain_important_terms:
                filtered_tokens.append(token)
            # Keep numbers and error codes
            elif re.match(r"^\d+$", token) or re.match(r"^[A-Z]\d+$", token):
                filtered_tokens.append(token)
            # Remove stop words
            elif token.lower() not in self.vietnamese_stop_words:
                filtered_tokens.append(token)

        return filtered_tokens

    def normalize_text_for_bm25(self, text: str) -> str:
        """Enhanced Vietnamese normalization for BM25"""
        if not text:
            return ""

        # Basic normalization
        normalized = text.lower().strip()

        # Vietnamese tokenization
        tokens = self.vietnamese_tokenize(normalized)

        # Remove stop words but keep domain terms
        tokens = self.remove_stop_words(tokens)

        # Filter out very short tokens (but keep numbers and codes)
        tokens = [
            token
            for token in tokens
            if len(token) >= 2
            or token.isdigit()
            or re.match(r"^[A-Z]\d+$", token.upper())
        ]

        # Join back
        normalized = " ".join(tokens)

        return normalized

    def bm25_tokenizer(self, text: str) -> str:
        if not text:
            return ""

        # Basic normalization
        normalized = text.lower().strip()

        # Vietnamese tokenization
        tokens = self.vietnamese_tokenize(normalized)

        # Remove stop words but keep domain terms
        tokens = self.remove_stop_words(tokens)

        # Filter out very short tokens (but keep numbers and codes)
        tokens = [
            token
            for token in tokens
            if len(token) >= 2
            or token.isdigit()
            or re.match(r"^[A-Z]\d+$", token.upper())
        ]

        return tokens

    def extract_error_code_variations(self, error_code: str) -> str:
        """Generate variations of error codes for better BM25 matching"""
        if not error_code:
            return ""

        variations = [error_code]

        # Add common Vietnamese variations
        if error_code.isdigit():
            # For numeric codes like "606"
            variations.extend(
                [
                    f"lỗi {error_code}",
                    f"error {error_code}",
                    f"mã {error_code}",
                    f"code {error_code}",
                    f"mã lỗi {error_code}",
                ]
            )
        else:
            # For alphanumeric codes like "W02", "BL2"
            variations.extend(
                [
                    f"lỗi {error_code}",
                    f"error {error_code}",
                    f"mã lỗi {error_code}",
                    f"code {error_code}",
                ]
            )

        return " ".join(variations)

    def extract_steps_keywords(self, guide_text: str) -> str:
        """Extract step-related keywords from procedure text"""
        if not guide_text:
            return ""

        # Find step patterns
        steps = re.findall(r"(?:bước|b)\s*\d+", guide_text, re.IGNORECASE)
        step_keywords = " ".join(steps)

        # Add common procedure keywords
        procedure_keywords = (
            "step bước instruction hướng dẫn guide quy trình process thao tác action"
        )

        return f"{step_keywords} {procedure_keywords}"

    def clean_column_name(self, column_name: str) -> str:
        """Clean column names by removing extra whitespace and newlines"""
        if not column_name:
            return ""

        # Remove newlines and extra spaces
        cleaned = re.sub(r"\s+", " ", column_name.strip())

        return cleaned
