import re
from typing import List, Dict
from markitdown import MarkItDown
from langchain.schema import Document
from src.processor.text_utils import VietnameseTextProcessor


class WordDocumentProcessor:
    """Process Word document content with semantic chunking using MarkItDown"""

    def __init__(self):
        self.text_processor = VietnameseTextProcessor()
        self.md_converter = MarkItDown()

        # Section patterns for Vietnamese content
        self.section_patterns = {
            "policy": r"(?:quy định|chính sách|policy)",
            "procedure": r"(?:hướng dẫn|cách|thao tác|bước)",
            "scope": r"(?:phạm vi|áp dụng|scope)",
            "fee": r"(?:phí|chiết khấu|fee|discount)",
            "timeout": r"(?:timeout|treo|đang xử lý|processing)",
        }

    def process_word_document(self, file_path: str) -> List[Document]:
        """Process Word document with semantic section chunking using MarkItDown"""
        try:
            # Convert document using MarkItDown
            result = self.md_converter.convert(file_path)
            content = result.text_content
            print(f"Document converted successfully. Content length: {len(content)}")
            print(f"First 500 characters:\n{content[:500]}...")

        except Exception as e:
            print(f"Error converting document with MarkItDown: {e}")
            return []

        documents = []

        # Extract semantic sections
        sections = self._extract_semantic_sections(content)

        for section in sections:
            processed_document = self._process_section(section, file_path)
            if processed_document:
                documents.append(processed_document)

        return documents

    def _extract_semantic_sections(self, content: str) -> List[Dict]:
        """Extract meaningful sections from Markdown content produced by MarkItDown"""
        sections = []

        # Since MarkItDown produces proper Markdown, we can use more sophisticated parsing
        # Split by headers (## or # followed by text)
        header_pattern = r"\n(?=#{1,6}\s+)"
        major_sections = re.split(header_pattern, content)

        for section_text in major_sections:
            if len(section_text.strip()) < 30:  # Skip very short sections
                continue

            section_info = self._analyze_markdown_section(section_text)
            if section_info:
                sections.append(section_info)

        # If no clear sections found, create chunks from paragraphs
        if not sections:
            sections = self._fallback_paragraph_chunking(content)

        return sections

    def _analyze_markdown_section(self, section_text: str) -> Dict:
        """Analyze Markdown section to determine type and extract content"""
        lines = section_text.strip().split("\n")

        if not lines:
            return None

        # Find the header line (starts with #)
        header = ""
        body_start_idx = 0

        for i, line in enumerate(lines):
            if line.strip().startswith("#"):
                header = line.strip()
                body_start_idx = i + 1
                break

        # If no header found, use first line as header
        if not header and lines:
            header = lines[0].strip()
            body_start_idx = 1

        # Get body content (rest of lines)
        body_lines = lines[body_start_idx:] if body_start_idx < len(lines) else []
        body = "\n".join(body_lines).strip()

        # Skip if body is too short or just references
        if len(body) < 20 or (
            len(body.split()) < 10
            and any(word in body.lower() for word in ["csv", "file", "document"])
        ):
            return None

        # Determine section type
        section_type = self._determine_section_type(header + " " + body)

        return {
            "header": header,
            "body": body,
            "type": section_type,
            "full_text": section_text.strip(),
        }

    def _fallback_paragraph_chunking(self, content: str) -> List[Dict]:
        """Fallback method to chunk content by paragraphs when no clear sections are found"""
        paragraphs = [
            p.strip()
            for p in content.split("\n\n")
            if p.strip() and len(p.strip()) > 50
        ]

        sections = []
        for i, paragraph in enumerate(paragraphs):
            # Use first line or sentence as header
            sentences = paragraph.split(".")
            header = sentences[0] if sentences else f"Section {i+1}"

            section_type = self._determine_section_type(paragraph)

            sections.append(
                {
                    "header": header,
                    "body": paragraph,
                    "type": section_type,
                    "full_text": paragraph,
                }
            )

        return sections

    def _determine_section_type(self, text: str) -> str:
        """Determine section type based on content"""
        text_lower = text.lower()

        for section_type, pattern in self.section_patterns.items():
            if re.search(pattern, text_lower):
                return section_type

        # Default to general content
        return "general"

    def _process_section(self, section: Dict, source_file: str) -> Document:
        """Process individual section into chunk, preserving Markdown structure"""
        header = section["header"]
        body = section["body"]
        section_type = section["type"]

        # Clean and format content
        if header.startswith("#"):
            # For Markdown headers, clean them up but preserve structure
            clean_header = re.sub(r"^#+\s*", "", header).strip()
            # Keep the structure but make it cleaner for content
            content = f"{clean_header}\n\n{body}".strip()
        else:
            content = f"{header}\n\n{body}".strip()

        # Remove excessive whitespace and normalize
        content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)  # Remove triple+ newlines
        content = re.sub(r"[ \t]+", " ", content)  # Normalize spaces and tabs

        return Document(
            page_content=content,
            metadata={
                "doc_type": section_type,
                "section_header": clean_header if header.startswith("#") else header,
                "category": "word_document",
                "has_markdown": (
                    "yes"
                    if any(marker in content for marker in ["#", "*", "|", "```"])
                    else "no"
                ),
                "content_length": len(content),
                "source_file": source_file,
            },
        )
