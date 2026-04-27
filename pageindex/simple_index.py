import PyPDF2
import os

def simple_index(pdf_path):
    pages = []

    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)

        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""

            pages.append({
                "page": i,
                "content": text
            })

    return {
        "doc_name": os.path.basename(pdf_path),
        "pages": pages
    }