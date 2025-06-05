import docx2txt

# Replace 'your_file.docx' with the path to your Word file
text = docx2txt.process("viettelpay_docs/raw/Nghiệp vụ.docx")

print(text[:2000])
