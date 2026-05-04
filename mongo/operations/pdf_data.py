from mongo.init import pdfs
from mongo.schema import PDFData

def insert_pdf(input_pdf_data: PDFData):
    # check for existing pdfs
    existing_pdf = pdfs.find_one({"doc_name": input_pdf_data.doc_name})

    if existing_pdf:
        # if pdf exists, return doc id
        return {
            "status": 200,
            "doc_id": str(existing_pdf["_id"])
        }

    else:
        # create a new pdf pydantic object
        new_pdf = PDFData(doc_name=input_pdf_data.doc_name, type=input_pdf_data.type, page_count=input_pdf_data.page_count, pages=input_pdf_data.pages)

        # insert new pdf and get the doc id
        new_pdf_data = pdfs.insert_one(new_pdf)

        # return app data
        return {
            "status": 201,
            "doc_id": new_pdf_data.inserted_id
        }

def fetch_pdf_pages(doc_id: int):
    # fetch the target pdf
    target_pdf = pdfs.find_one({"_id": doc_id})

    if not target_pdf:
         # if target pdf not present return error
        return {
            "status": 404,
            "message": "PDF not found"
        }
    else:
        # return pages or whatever bm25 needs
        return {
            "status": 200,
            "pages": target_pdf["pages"]
        }