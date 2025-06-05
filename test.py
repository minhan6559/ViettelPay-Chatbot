from docx2python import docx2python

# Extract with structure
doc = docx2python("viettelpay_docs/raw/Nghiệp vụ.docx")


def extract_cell_content(cell):
    """Extract actual text content from cell (which might be a list)"""
    if isinstance(cell, list):
        # If it's a list, join all non-empty items
        content_parts = []
        for item in cell:
            if isinstance(item, str) and item.strip():
                content_parts.append(item.strip())
            elif isinstance(item, list):
                # Handle nested lists recursively
                nested_content = extract_cell_content(item)
                if nested_content:
                    content_parts.append(nested_content)
        return "\n".join(content_parts) if content_parts else ""
    elif isinstance(cell, str):
        return cell.strip()
    else:
        return str(cell).strip() if cell else ""


def format_table(table):
    """Format table based on number of cells per row"""
    formatted_rows = []

    for row in table:
        # Extract actual content from each cell and filter out empty ones
        processed_cells = []
        for cell in row:
            content = extract_cell_content(cell)
            if content:  # Only add non-empty content
                processed_cells.append(content)

        if len(processed_cells) <= 1:
            # Single cell or empty row - join with newline
            # This is a paragraph
            if processed_cells:
                formatted_rows.append(processed_cells[0])
        else:
            # Multiple cells - join with " | "
            # This is a table
            formatted_rows.append(" | ".join(processed_cells))

    return "\n".join(formatted_rows)


def format_document(doc):
    """Format the entire document with proper table formatting"""
    formatted_content = []

    # Process the body content
    for i, table in enumerate(doc.body):
        if table:  # Skip empty tables
            formatted_table = format_table(table)
            if formatted_table.strip():  # Only add non-empty content
                formatted_content.append(f"=== Table/Section {i+1} ===")
                formatted_content.append(formatted_table)
                formatted_content.append("")  # Add spacing between sections

    return "\n".join(formatted_content)


# Format and print the document
formatted_doc = format_document(doc)
# print(formatted_doc)

print("\n" + "=" * 80)
print("ORIGINAL TEXT CONTENT (first 500 chars):")
print(doc.text[:2000])

# print("-" * 100)
# for document in doc.document:
#     print(document)
