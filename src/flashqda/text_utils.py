import re
import pysbd
from pysbd.lang.common.standard import Standard

# Extend pySBD's English abbreviation list with terms common in academic text.
# 'approx', 'ca', 'vol', 'ed', 'eds' are absent from the default list.
_EXTRA_ABBREVS = ["approx", "ca", "vol", "ed", "eds"]
for _abbr in _EXTRA_ABBREVS:
    if _abbr not in Standard.Abbreviation.ABBREVIATIONS:
        Standard.Abbreviation.ABBREVIATIONS.append(_abbr)


def _normalize_line_breaks(text: str) -> str:
    """
    Normalize PDF-extraction line break artifacts before segmentation.
    - Joins hyphenated line breaks: "hyphen-\\nated" -> "hyphenated"
    - Collapses single newlines to spaces, preserving blank-line paragraph markers.
    """
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    return text


def segment_sentences(text, custom_items=None):
    """
    Segment text into sentences using pySBD (Pragmatic Sentence Boundary Disambiguation).

    pySBD correctly handles academic abbreviations (Dr., et al., Fig., ca., approx.),
    decimal notation, numbered items, and quoted speech. Single newlines from PDF
    line-wrapping are normalised to spaces before segmentation so output sentences
    contain no embedded newline characters.

    Args:
        text (str): Input document text.
        custom_items: Deprecated. Retained for backward compatibility; has no effect.

    Returns:
        list[str]: List of sentence strings, free of embedded newlines.
    """
    text = text.replace(' ', ' ')
    text = _normalize_line_breaks(text)
    segmenter = pysbd.Segmenter(language='en', clean=False)
    sentences = segmenter.segment(text)
    return [s.strip() for s in sentences if s.strip()]


def segment_paragraphs(text):
    """
    Segment text into paragraphs by splitting on blank lines.

    Within each paragraph, single newlines (PDF line-wrap artifacts) are collapsed
    to spaces so output paragraphs contain no embedded newline characters.

    Known limitation: PDFs in which paragraphs are separated by only a single
    newline (no blank line) in the extracted text cannot be reliably segmented.
    Use granularity="sentence" (the default) for PDF files.

    Args:
        text (str): Input document text.

    Returns:
        list[str]: List of paragraph strings, free of embedded newlines.
    """
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    paragraphs = re.split(r'\n{2,}', text)
    result = []
    for p in paragraphs:
        p = _normalize_line_breaks(p).strip()
        if p:
            result.append(p)
    return result
