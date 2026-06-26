"""Tests for the public API surface exported by flashqda.__init__."""
import flashqda


def test_all_exports_are_importable():
    for name in flashqda.__all__:
        assert hasattr(flashqda, name), f"'{name}' is in __all__ but not importable from flashqda"


def test_explode_extractions_correct_spelling():
    assert hasattr(flashqda, "explode_extractions"), "explode_extractions must be importable"
    assert not hasattr(flashqda, "expode_extractions"), "typo 'expode_extractions' should not exist"


def test_refine_extracted_in_all():
    assert "refine_extracted" in flashqda.__all__
    assert hasattr(flashqda, "refine_extracted")


def test_core_functions_present():
    expected = [
        "initialize_project",
        "ProjectContext",
        "preprocess_documents",
        "PipelineConfig",
        "link_items",
        "get_llm_api_key",
        "classify_items",
        "label_items",
        "extract_from_classified",
        "refine_extracted",
        "embed_items",
        "group_items",
        "remap_from_categories_csv",
        "explode_extractions",
    ]
    for name in expected:
        assert name in flashqda.__all__, f"'{name}' missing from __all__"
        assert hasattr(flashqda, name), f"'{name}' not importable"
