from pathlib import Path


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ImportError(
            "Reading PDF files requires pymupdf. "
            "Install it with: pip install pymupdf  or  pip install 'flashqda[preprocessing]'"
        )
    doc = fitz.open(str(path))
    page_texts = []
    for page in doc:
        page_blocks = page.get_text("blocks")
        # block tuple: (x0, y0, x1, y1, text, block_no, block_type)
        # block_type 0 = text; skip image blocks (type 1)
        text_blocks = [
            b[4].strip()
            for b in sorted(page_blocks, key=lambda b: (b[1], b[0]))
            if b[6] == 0 and b[4].strip()
        ]
        if text_blocks:
            page_texts.append("\n\n".join(text_blocks))
    return "\n\n".join(page_texts)


def _read_docx(path: Path) -> str:
    try:
        import docx
    except ImportError:
        raise ImportError(
            f"Reading DOCX files requires python-docx. "
            "Install it with: pip install python-docx  or  pip install 'flashqda[preprocessing]'"
        )
    doc = docx.Document(path)
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


_READERS = {
    ".txt": _read_txt,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
}


def get_documents(project):
    """
    Collect documents from the data folder.

    Supported formats: .txt, .pdf (requires pdfplumber), .docx (requires python-docx).
    Files with unrecognised extensions are skipped with a printed warning.
    Files are processed in alphabetical order for deterministic document_id assignment.
    """
    data_dir = Path(project.data)
    documents = []

    for entry in sorted(data_dir.iterdir()):
        if not entry.is_file():
            continue
        ext = entry.suffix.lower()
        if ext not in _READERS:
            print(f"[WARN] Skipping unsupported file: {entry.name}")
            continue
        reader = _READERS[ext]
        text = reader(entry)
        documents.append({"filename": entry.name, "text": text})

    return documents
