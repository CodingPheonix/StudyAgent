from typing import List

from pydantic import BaseModel


class User(BaseModel):
    name: str
    email: str
    password: str

class Page(BaseModel):
    page: int
    summary: str
    content: str

class PDFData(BaseModel):
    type: str
    doc_name: str
    page_count: int
    pages: List[Page]