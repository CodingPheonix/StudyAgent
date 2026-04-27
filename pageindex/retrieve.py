import json
import PyPDF2
import math
from collections import Counter, defaultdict

try:
    from .utils import get_number_of_pages, remove_fields
except ImportError:
    from utils import get_number_of_pages, remove_fields

class BM25:

    def __init__(self, docs):
        self.docs = docs
        self.doc_len = []
        self.avgdl = 0
        self.freqs = []
        self.idf = {}
        self.k1 = 1.5
        self.b = 0.75

        self._initialize()

    def _initialize(self):
        df = defaultdict(int)
        total_len = 0

        for doc in self.docs:
            tokens = doc.split()
            total_len += len(tokens)
            self.doc_len.append(len(tokens))

            freq = Counter(tokens)
            self.freqs.append(freq)

            for word in freq:
                df[word] += 1

        self.avgdl = total_len / len(self.docs)

        for word, count in df.items():
            self.idf[word] = math.log(1 + (len(self.docs) - count + 0.5) / (count + 0.5))

    def score(self, query, index):
        score = 0
        tokens = query.split()
        freq = self.freqs[index]
        dl = self.doc_len[index]

        for word in tokens:
            if word not in freq:
                continue

            f = freq[word]
            idf = self.idf.get(word, 0)

            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (f * (self.k1 + 1)) / denom

        return score


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_pages(pages: str) -> list[int]:
    """Parse a pages string like '5-7', '3,8', or '12' into a sorted list of ints."""
    result = []
    for part in pages.split(','):
        part = part.strip()
        if '-' in part:
            start, end = int(part.split('-', 1)[0].strip()), int(part.split('-', 1)[1].strip())
            if start > end:
                raise ValueError(f"Invalid range '{part}': start must be <= end")
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def _count_pages(doc_info: dict) -> int:
    """Return total page count for a PDF document."""
    if doc_info.get('page_count'):
        return doc_info['page_count']
    if doc_info.get('pages'):
        return len(doc_info['pages'])
    return get_number_of_pages(doc_info['path'])


def _get_pdf_page_content(doc_info: dict, page_nums: list[int]) -> list[dict]:
    """Extract text for specific PDF pages (1-indexed). Prefer cached pages, fallback to PDF."""
    cached_pages = doc_info.get('pages')
    if cached_pages:
        page_map = {p['page']: p['content'] for p in cached_pages}
        return [
            {'page': p, 'content': page_map[p]}
            for p in page_nums if p in page_map
        ]
    path = doc_info['path']
    with open(path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        total = len(pdf_reader.pages)
        valid_pages = [p for p in page_nums if 1 <= p <= total]
        return [
            {'page': p, 'content': pdf_reader.pages[p - 1].extract_text() or ''}
            for p in valid_pages
        ]


def _get_md_page_content(doc_info: dict, page_nums: list[int]) -> list[dict]:
    """
    For Markdown documents, 'pages' are line numbers.
    Find nodes whose line_num falls within [min(page_nums), max(page_nums)] and return their text.
    """
    min_line, max_line = min(page_nums), max(page_nums)
    results = []
    seen = set()

    def _traverse(nodes):
        for node in nodes:
            ln = node.get('line_num')
            if ln and min_line <= ln <= max_line and ln not in seen:
                seen.add(ln)
                results.append({'page': ln, 'content': node.get('text', '')})
            if node.get('nodes'):
                _traverse(node['nodes'])

    _traverse(doc_info.get('structure', []))
    results.sort(key=lambda x: x['page'])
    return results


# ── Tool functions ────────────────────────────────────────────────────────────

def get_document(documents: dict, doc_id: str) -> str:
    """Return JSON with document metadata: doc_id, doc_name, doc_description, type, status, page_count (PDF) or line_count (Markdown)."""
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})
    result = {
        'doc_id': doc_id,
        'doc_name': doc_info.get('doc_name', ''),
        'doc_description': doc_info.get('doc_description', ''),
        'type': doc_info.get('type', ''),
        'status': 'completed',
    }
    if doc_info.get('type') == 'pdf':
        result['page_count'] = _count_pages(doc_info)
    else:
        result['line_count'] = doc_info.get('line_count', 0)
    return json.dumps(result)


# def get_document_structure(documents: dict, doc_id: str) -> str:
#     """Return tree structure JSON with text fields removed (saves tokens)."""
#     doc_info = documents.get(doc_id)
#     if not doc_info:
#         return json.dumps({'error': f'Document {doc_id} not found'})
#     structure = doc_info.get('structure', [])
#     structure_no_text = remove_fields(structure, fields=['text'])
#     return json.dumps(structure_no_text, ensure_ascii=False)

def get_document_structure(documents: dict, doc_id: str) -> str:
    doc_info = documents.get(doc_id)

    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})

    pages = doc_info.get("pages", [])

    return json.dumps({
        "doc_name": doc_info.get("doc_name", ""),
        "total_pages": len(pages),
        "preview_pages": [p["page"] for p in pages[:5]]
    }, ensure_ascii=False)

# def get_page_content(documents: dict, doc_id: str, pages: str) -> str:
#     """
#     Retrieve page content for a document.

#     pages format: '5-7', '3,8', or '12'
#     For PDF: pages are physical page numbers (1-indexed).
#     For Markdown: pages are line numbers corresponding to node headers.

#     Returns JSON list of {'page': int, 'content': str}.
#     """
#     doc_info = documents.get(doc_id)
#     if not doc_info:
#         return json.dumps({'error': f'Document {doc_id} not found'})

#     try:
#         page_nums = _parse_pages(pages)
#     except (ValueError, AttributeError) as e:
#         return json.dumps({'error': f'Invalid pages format: {pages!r}. Use "5-7", "3,8", or "12". Error: {e}'})

#     try:
#         if doc_info.get('type') == 'pdf':
#             content = _get_pdf_page_content(doc_info, page_nums)
#         else:
#             content = _get_md_page_content(doc_info, page_nums)
#     except Exception as e:
#         return json.dumps({'error': f'Failed to read page content: {e}'})

#     return json.dumps(content, ensure_ascii=False)

def get_page_content(documents: dict, doc_id: str, pages: str) -> str:
    """
    Retrieve page content for a document.

    pages format: '5-7', '3,8', or '12'
    Returns JSON list of {'page': int, 'content': str}.
    """

    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})

    try:
        page_nums = _parse_pages(pages)
    except (ValueError, AttributeError) as e:
        return json.dumps({
            'error': f'Invalid pages format: {pages!r}. Use "5-7", "3,8", or "12". Error: {e}'
        })

    # 🔥 SAFETY LIMIT (VERY IMPORTANT)
    if len(page_nums) > 5:
        return json.dumps({'error': 'Too many pages requested (max 5 allowed)'})

    try:
        if doc_info.get('type') == 'pdf':
            content = _get_pdf_page_content(doc_info, page_nums)
        else:
            content = _get_md_page_content(doc_info, page_nums)
    except Exception as e:
        return json.dumps({'error': f'Failed to read page content: {e}'})

    # 🔥 CLEAN EMPTY PAGES
    content = [c for c in content if c.get("content", "").strip()]

    return json.dumps(content, ensure_ascii=False)

# def search_pages(doc: dict, query: str, top_k: int = 2):
#     pages = doc.get("pages", [])
#     query_words = query.lower().split()

#     scores = []

#     for page in pages:
#         text = page.get("content", "").lower()
#         score = sum(text.count(word) for word in query_words)
#         scores.append((score, page.get("page")))

#     scores.sort(reverse=True)

#     top_pages = [p for score, p in scores if score > 0][:top_k]

#     # 🔥 controlled expansion
#     expanded = set()
#     for p in top_pages:
#         expanded.add(p)
#         if p > 1:
#             expanded.add(p - 1)
#         expanded.add(p + 1)

#     # 🔥 LIMIT TO MAX 5 PAGES
#     return sorted(list(expanded))[:5]

def search_pages(doc: dict, query: str, top_k: int = 3):

    pages = doc.get("pages", [])
    texts = [p.get("content", "").lower() for p in pages]

    if not texts:
        return []

    bm25 = BM25(texts)

    scores = []
    query = query.lower()

    for i in range(len(texts)):
        score = bm25.score(query, i)
        scores.append((score, pages[i]["page"]))

    scores.sort(reverse=True)

    top_pages = [p for score, p in scores if score > 0][:top_k]

    # 🔥 controlled expansion (context)
    expanded = set()
    for p in top_pages:
        expanded.add(p)
        if p > 1:
            expanded.add(p - 1)
        expanded.add(p + 1)

    return sorted(list(expanded))[:5]

def smart_get_content(documents: dict, doc_id: str, query: str) -> str:
    doc = documents.get(doc_id)

    if not doc:
        return json.dumps({'error': 'Document not found'})

    pages = search_pages(doc, query)

    if not pages:
        return json.dumps({'error': 'No relevant content found'})

    pages_str = ",".join(map(str, pages[:5]))

    return get_page_content(documents, doc_id, pages_str)