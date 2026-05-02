import PyPDF2
import os
import spacy
import pytextrank
from PIL import Image
import pytesseract
import fitz

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")


def clean_pdf_text(text: str) -> str:
    return " ".join((text or "").split())


def simple_index(pdf_path):
    pages = []

    # doc = fitz.open(pdf_path)
    #
    # for page in doc:
    #     pix = page.get_pixmap()
    #     img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    #     text = pytesseract.image_to_string(img)
    #     print(text)

    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)

        for i, page in enumerate(reader.pages, 1):
            text = clean_pdf_text(page.extract_text() or "")

            doc = nlp(text)
            summary = clean_pdf_text(" ".join([
                sent.text for sent in doc._.textrank.summary(limit_phrases=5, limit_sentences=3)
            ]))

            pages.append({
                "page": i,
                "content": text,
                "summary": summary
            })


    return {
        "doc_name": os.path.basename(pdf_path),
        "pages": pages
    }
