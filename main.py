import os.path

from pathlib import Path

import gdown
import requests
from aiohttp.web_response import json_response
from fastapi import FastAPI
from pydantic import BaseModel

from pageindex import PageIndexClient

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "documents"
WORKSPACE = BASE_DIR / "workspace"

class PdfUrlInput(BaseModel):
    url: str
    filename: str

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}

def convert_drive_url(url: str) -> str:
    """Convert Google Drive share URL to direct download URL"""
    if "drive.google.com" in url:
        file_id = url.split("/d/")[1].split("/")[0]
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url


@app.post("/download_pdf")
def download_pdf(params: PdfUrlInput):
    print(params.url)

    if not params.url or not params.filename:
        return json_response({"message": "URL or Filename not provided", "status": 404})

    file_path_str = os.path.join(PDF_DIR, params.filename)
    file_path = Path(file_path_str)

    if "drive.google.com" in params.url:
        import gdown
        gdown.download(params.url, str(file_path), quiet=False)

    else:
        response = requests.get(params.url, stream=True, timeout=30)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(8192):
                if chunk:
                    f.write(chunk)

    # ✅ Validate PDF
    with open(file_path, "rb") as f:
        if not f.read(5).startswith(b"%PDF"):
            raise ValueError("Downloaded file is not a valid PDF")

    client = PageIndexClient(workspace=WORKSPACE)

    print("=" * 60)
    print("Step 1: Index PDF and view tree structure")
    print("=" * 60)
    doc_id = next(
        (did for did, doc in client.documents.items() if doc.get("doc_name") == file_path.name),
        None,
    )
    if doc_id:
        print(f"\nLoaded cached doc_id: {doc_id}")
    else:
        doc_id = client.index(file_path_str)
        print(f"\nIndexed. doc_id: {doc_id}")

    print("\nDocument Info:")
    structure = client.get_document_structure(doc_id)
    print(structure)

    # Delete File chunk
    if file_path.exists():
        file_path.unlink()
    print("Deleted")

    return structure