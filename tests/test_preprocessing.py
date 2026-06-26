"""Tests for document_io.py and text_utils.py."""
import sys
import types
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from flashqda.text_utils import segment_sentences, segment_paragraphs


class TestSegmentSentences:
    def test_basic_split(self):
        text = "The sky is blue. Clouds appear in the morning. Rain follows."
        result = segment_sentences(text)
        assert len(result) == 3

    def test_abbreviation_not_split(self):
        text = "Mr. Smith visited Dr. Jones on Monday. They discussed the findings."
        result = segment_sentences(text)
        assert len(result) == 2

    def test_returns_stripped_strings(self):
        text = "  First sentence.  Second sentence.  "
        result = segment_sentences(text)
        for s in result:
            assert s == s.strip()

    def test_non_breaking_space_normalised(self):
        text = "First sentence. Second sentence."
        result = segment_sentences(text)
        assert all(" " not in s for s in result)

    def test_custom_items_ignored_backward_compat(self):
        text = "One sentence. Two sentences."
        result_default = segment_sentences(text)
        result_custom = segment_sentences(text, custom_items=["abbr."])
        assert result_default == result_custom

    def test_empty_string(self):
        result = segment_sentences("")
        assert result == []


class TestSegmentParagraphs:
    def test_splits_on_blank_line(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = segment_paragraphs(text)
        assert len(result) == 3

    def test_does_not_split_within_paragraph(self):
        text = "Line one of paragraph.\nLine two of paragraph.\n\nSecond paragraph."
        result = segment_paragraphs(text)
        assert len(result) == 2
        assert "Line one" in result[0]
        assert "Line two" in result[0]

    def test_windows_line_endings_normalised(self):
        text = "First paragraph.\r\n\r\nSecond paragraph."
        result = segment_paragraphs(text)
        assert len(result) == 2

    def test_old_mac_line_endings_normalised(self):
        text = "First paragraph.\r\rSecond paragraph."
        result = segment_paragraphs(text)
        assert len(result) == 2

    def test_empty_paragraphs_filtered(self):
        text = "First.\n\n\n\nSecond."
        result = segment_paragraphs(text)
        assert len(result) == 2

    def test_empty_string(self):
        assert segment_paragraphs("") == []


class TestGetDocuments:
    def test_reads_txt_files(self, tmp_path):
        from flashqda.document_io import get_documents
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "doc1.txt").write_text("Hello world.", encoding="utf-8")
        project = MagicMock()
        project.data = tmp_path / "data"
        docs = get_documents(project)
        assert len(docs) == 1
        assert docs[0]["filename"] == "doc1.txt"
        assert "Hello world." in docs[0]["text"]

    def test_alphabetical_ordering(self, tmp_path):
        from flashqda.document_io import get_documents
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "b.txt").write_text("B doc.", encoding="utf-8")
        (tmp_path / "data" / "a.txt").write_text("A doc.", encoding="utf-8")
        project = MagicMock()
        project.data = tmp_path / "data"
        docs = get_documents(project)
        assert docs[0]["filename"] == "a.txt"
        assert docs[1]["filename"] == "b.txt"

    def test_unsupported_extension_warns(self, tmp_path, capsys):
        from flashqda.document_io import get_documents
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "notes.odt").write_text("ignored", encoding="utf-8")
        project = MagicMock()
        project.data = tmp_path / "data"
        docs = get_documents(project)
        assert docs == []
        captured = capsys.readouterr()
        assert "notes.odt" in captured.out

    def test_pdf_raises_without_pymupdf(self, tmp_path):
        from flashqda.document_io import _read_pdf
        fake_path = tmp_path / "doc.pdf"
        fake_path.write_bytes(b"%PDF-1.4")
        with patch.dict(sys.modules, {"fitz": None}):
            with pytest.raises(ImportError, match="pymupdf"):
                _read_pdf(fake_path)

    def test_pdf_blocks_join_with_blank_lines(self, tmp_path):
        from flashqda.document_io import _read_pdf
        from unittest.mock import MagicMock
        fake_path = tmp_path / "doc.pdf"
        fake_path.write_bytes(b"%PDF-1.4")

        mock_page = MagicMock()
        mock_page.get_text.return_value = [
            (0, 0, 200, 50, "First paragraph text.", 0, 0),
            (0, 60, 200, 110, "Second paragraph text.", 1, 0),
        ]
        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            result = _read_pdf(fake_path)

        assert "First paragraph text." in result
        assert "Second paragraph text." in result
        assert "\n\n" in result

    def test_docx_raises_without_python_docx(self, tmp_path):
        from flashqda.document_io import _read_docx
        fake_path = tmp_path / "doc.docx"
        fake_path.write_bytes(b"PK")
        with patch.dict(sys.modules, {"docx": None}):
            with pytest.raises(ImportError, match="python-docx"):
                _read_docx(fake_path)


class TestSegmentSentencesPySBD:
    def test_abbreviations_not_split(self):
        # pySBD handles Dr., Mr., Mrs., Fig., et al.; approx. patched via extra_abbreviations
        text = "Dr. Smith and et al. confirmed that approx. 80% of cases showed improvement."
        result = segment_sentences(text)
        assert len(result) == 1

    def test_figure_reference_not_split(self):
        text = "As shown in Fig. 2, the correlation is strong. Results confirm this."
        result = segment_sentences(text)
        assert len(result) == 2

    def test_decimal_not_split(self):
        # decimal within a sentence should not be treated as a sentence end
        text = "The p-value was 0.05. This indicates significance."
        result = segment_sentences(text)
        assert len(result) == 2

    def test_no_embedded_newlines(self):
        text = "A sentence that wraps\nacross lines in the PDF. Second sentence."
        result = segment_sentences(text)
        for s in result:
            assert '\n' not in s

    def test_hyphenated_join(self):
        text = "A hyphen-\nated word appears here. Next sentence."
        result = segment_sentences(text)
        assert 'hyphenated' in result[0]
        assert '\n' not in result[0]

    def test_wrapped_sentence_kept_together(self):
        text = "This is a long sentence that wraps\nacross multiple lines in the PDF."
        result = segment_sentences(text)
        assert len(result) == 1


class TestSegmentParagraphsNormalization:
    def test_no_embedded_newlines_in_paragraphs(self):
        text = "Para one line one\nline two of para one.\n\nPara two."
        result = segment_paragraphs(text)
        for p in result:
            assert '\n' not in p

    def test_hyphenated_join_in_paragraph(self):
        text = "A hyphen-\nated word.\n\nSecond paragraph."
        result = segment_paragraphs(text)
        assert 'hyphenated' in result[0]
        assert '\n' not in result[0]
