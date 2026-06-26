# flashqda/__init__.py

__all__ = [
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
    "estimate_cost",
]

# Expose functions at the package level
from .project_setup import initialize_project
from .project_context import ProjectContext
from .preprocess import preprocess_documents
from .pipelines.config import PipelineConfig
from .causal_chain import link_items
from .llm_utils import get_llm_api_key
from .pipeline_runner import classify_items, label_items, extract_from_classified, refine_extracted, explode_extractions
from .embedding_pipeline import embed_items

def group_items(*args, **kwargs):
    """Run AHC-based semantic grouping on a CSV file. See ``flashqda.grouping.pipeline.group_items`` for full parameter docs."""
    from .grouping.pipeline import group_items as _group_items
    return _group_items(*args, **kwargs)

def remap_from_categories_csv(*args, **kwargs):
    """Re-apply a manually edited categories CSV to the original dataframe. See ``flashqda.grouping.mapper.remap_from_categories_csv`` for full parameter docs."""
    from .grouping.mapper import remap_from_categories_csv as _remap
    return _remap(*args, **kwargs)

from .cost_estimation import estimate_cost