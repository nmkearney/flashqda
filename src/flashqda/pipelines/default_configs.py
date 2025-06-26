# default_configs.py — maps pipeline_type to defaults

PIPELINE_CONFIGS = {
    "causal": {
        "labels": ["causal", "non-causal"],
        "extract_labels": ["cause", "effect"],
        "prompt_files": {
            "classify": "causal_classify.txt",
            "label_abstract": "label_abstract.txt",
            "label_sent_para": "label_sent_para.txt",
            "label_extracted": "label_extracted.txt",
            "extract": "causal_extract.txt",
        },
        "system_prompt": "You are helping identify causal relationships. Respond using JSON."
    }
}