# default_configs.py — maps pipeline_type to defaults

PIPELINE_CONFIGS = {
    "causal": {
        "classify_labels": ["causal", "non-causal"],
        "extract_fields": ["cause", "effect"],
        "prompt_files": {
            "classify": "causal_classify.txt",
            "label_sent_para": "label_sent_para.txt",
            "label_extracted": "label_extracted.txt",
            "extract": "causal_extract.txt",
            "refine_extract": "causal_refine_extract.txt"
        },
        "system_prompt": "You are helping identify causal relationships.",
        "classify_schema": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "enum": ["causal", "non-causal"]
                }
            },
            "required": ["label"],
            "additionalProperties": False
        },
        "extract_schema": {
            "type": "object",
            "properties": {
                "relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cause": {"type": "string"},
                            "effect": {"type": "string"}
                        },
                        "required": ["cause", "effect"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["relationships"],
            "additionalProperties": False
        },
        "provider": "openai",
        "model": "gpt-4o",
        "base_url": None,
        "api_key": None,
        "temperature": 0.0,
        "timeout": 30,
        "use_json_mode": True,
    },
    "thematic": {
        "classify_labels": ["thematic", "non-thematic"],
        "extract_fields": ["theme", "evidence"],
        "prompt_files": {
            "label_sent_para": "label_sent_para.txt",
            "label_extracted": "label_extracted.txt",
            "extract":         "thematic_extract.txt",
            "refine_extract":  "thematic_refine_extract.txt",
        },
        "system_prompt": "You are helping identify themes in qualitative data.",
        "provider": "openai",
        "model": "gpt-4o",
        "base_url": None,
        "api_key": None,
        "temperature": 0.0,
        "timeout": 30,
        "use_json_mode": True,
    },
    "tradeoff": {
        "classify_labels": ["tradeoff", "non-tradeoff"],
        "extract_fields": ["gain", "loss"],
        "prompt_files": {
            "classify":        "tradeoff_classify.txt",
            "label_sent_para": "label_sent_para.txt",
            "label_extracted": "label_extracted.txt",
            "extract":         "tradeoff_extract.txt",
            "refine_extract":  "tradeoff_refine_extract.txt",
        },
        "system_prompt": "You are helping identify trade-offs and tensions.",
        "provider": "openai",
        "model": "gpt-4o",
        "base_url": None,
        "api_key": None,
        "temperature": 0.0,
        "timeout": 30,
        "use_json_mode": True,
    },
    "synergy": {
        "classify_labels": ["synergy", "non-synergy"],
        "extract_fields": ["factor_a", "factor_b"],
        "prompt_files": {
            "classify":        "synergy_classify.txt",
            "label_sent_para": "label_sent_para.txt",
            "label_extracted": "label_extracted.txt",
            "extract":         "synergy_extract.txt",
            "refine_extract":  "synergy_refine_extract.txt",
        },
        "system_prompt": "You are helping identify synergies and mutually reinforcing relationships.",
        "provider": "openai",
        "model": "gpt-4o",
        "base_url": None,
        "api_key": None,
        "temperature": 0.0,
        "timeout": 30,
        "use_json_mode": True,
    },
}