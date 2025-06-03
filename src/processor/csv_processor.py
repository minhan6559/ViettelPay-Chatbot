import pandas as pd
from typing import List
from langchain.schema import Document
from src.processor.text_utils import VietnameseTextProcessor


class CSVProcessor:
    """Process various CSV files from ViettelPay documentation"""

    def __init__(self):
        self.text_processor = VietnameseTextProcessor()

    def process_definitions(self, file_path: str) -> List[Document]:
        """Process dinh_nghia.csv"""
        df = pd.read_csv(file_path)
        documents = []

        for _, row in df.iterrows():
            term = str(row["Định nghĩa"]).strip()
            definition = str(row["Giải thích"]).strip()

            # Create content
            content = f"Định nghĩa - {term}: {definition}"

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "doc_type": "definition",
                        "term": term,
                        "category": "terminology",
                        "source_file": "dinh_nghia.csv",
                    },
                )
            )

        return documents

    def process_error_handling(self, file_path: str) -> List[Document]:
        """Process huong_dan_xu_ly_loi.csv - Most critical for customer support"""
        df = pd.read_csv(file_path)
        documents = []

        for _, row in df.iterrows():
            error_code = str(row["Mã lỗi"]).strip()
            error_msg = str(row["Câu báo lỗi/yêu cầu hỗ trợ"]).strip()
            service = str(
                row["Nghiệp vụ"]
            ).strip()  # Fixed column name (no trailing space)
            cause = str(row["Nguyên nhân"]).strip()
            status = str(row["Trạng thái giao dịch"]).strip()
            solution = str(row["Hướng khắc phục"]).strip()

            # Create comprehensive content
            content = f"""
Mã lỗi {error_code}: {error_msg}

Nghiệp vụ: {service}
Nguyên nhân: {cause}
Trạng thái: {status}

Cách khắc phục: {solution}
            """.strip()

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "doc_type": "error_resolution",
                        "error_code": error_code,
                        "service_type": service,
                        "status": status,
                        "source_file": "huong_dan_xu_ly_loi.csv",
                    },
                )
            )

        return documents

    def process_payment_guide(self, file_path: str) -> List[Document]:
        """Process huong_dan_thanh_toan.csv"""
        df = pd.read_csv(file_path)
        documents = []

        for _, row in df.iterrows():
            transaction_type = str(row["Loại giao dịch"]).strip()
            scope = str(row["Phạm vi thanh toán"]).strip()
            guide = str(row["Hướng dẫn thanh toán"]).strip()

            content = f"""
Hướng dẫn thực hiện giao dịch

Loại giao dịch: {transaction_type}

Phạm vi: {scope}

Các bước thực hiện:
{guide}
            """.strip()

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "doc_type": "procedure",
                        "transaction_type": transaction_type,
                        "service_scope": scope,
                        "category": "payment_guide",
                        "source_file": "huong_dan_thanh_toan.csv",
                    },
                )
            )

        return documents

    def process_error_codes(self, file_path: str) -> List[Document]:
        """Process bang_ma_loi.csv"""
        df = pd.read_csv(file_path)
        documents = []

        for _, row in df.iterrows():
            error_code = str(row["Mã lỗi"]).strip()
            description = str(row["Mô tả"]).strip()

            content = f"Mã lỗi {error_code}: {description}"

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "doc_type": "error_code",
                        "error_code": error_code,
                        "category": "error_reference",
                        "source_file": "bang_ma_loi.csv",
                    },
                )
            )

        return documents

    def process_cancellation_rules(self, file_path: str) -> List[Document]:
        """Process quy_dinh_huy_giao_dich.csv"""
        df = pd.read_csv(file_path)
        documents = []

        # File summary for better semantic context
        file_summary = """
Tài liệu quy định hủy giao dịch ViettelPay.
Bao gồm các quy định về điều kiện, hạn mức, nguyên tắc và hướng dẫn thực hiện hủy giao dịch thanh toán cước viễn thông Viettel.
        """.strip()

        for _, row in df.iterrows():
            rule_type = str(row["Nội dung"]).strip()
            rule_details = str(row["Quy định"]).strip()

            # Create content with summary for better semantic understanding
            content = f"""
{file_summary}

Nội dung: {rule_type}

Quy định: {rule_details}
            """.strip()

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "doc_type": "policy",
                        "rule_type": rule_type,
                        "category": "cancellation_rules",
                        "source_file": "quy_dinh_huy_giao_dich.csv",
                    },
                )
            )

        return documents

    def process_denominations(self, file_path: str) -> List[Document]:
        """Process menh_gia.csv - Concatenated approach for comparison queries"""
        df = pd.read_csv(file_path)

        # Create a summary chunk for denominations (concatenated approach)
        content = "Bảng mệnh giá thẻ cào theo nhà mạng:\n\n"

        for _, row in df.iterrows():
            denomination = str(
                row["Mệnh giá (Đơn vị tính VNĐ)"]
            ).strip()  # Fixed column name
            viettel = str(row["Nhà mạng Viettel"]).strip()
            mobifone = str(row["Nhà mạng Mobifone"]).strip()
            vinaphone = str(row["Nhà mạng Vinaphone"]).strip()
            vietnammobile = str(row["Nhà mạng Vietnammobile"]).strip()

            content += f"Mệnh giá {denomination}: Viettel({viettel}), Mobifone({mobifone}), Vinaphone({vinaphone}), Vietnammobile({vietnammobile})\n"

        return [
            Document(
                page_content=content,
                metadata={
                    "doc_type": "reference",
                    "category": "denominations",
                    "source_file": "menh_gia.csv",
                },
            )
        ]
