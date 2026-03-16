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
    "embed_items",
    "group_items"
]

# Expose functions at the package level
from .project_setup import initialize_project
from .project_context import ProjectContext
from .preprocess import preprocess_documents
from .pipelines.config import PipelineConfig
from .causal_chain import link_items
from .llm_utils import get_llm_api_key
from .pipeline_runner import classify_items, label_items, extract_from_classified
from .embedding_pipeline import embed_items

def group_items(*args, **kwargs):
    from .grouping.pipeline import group_items as _group_items
    return _group_items(*args, **kwargs)