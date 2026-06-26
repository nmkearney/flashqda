from pathlib import Path

DEFAULT_PROMPT_DIR = Path(__file__).parent / "prompts"
LOCAL_PROMPT_DIR = DEFAULT_PROMPT_DIR / "local"

_LOCAL_PROVIDERS = {"ollama", "openai_compatible"}


def _should_use_local_prompts(config) -> bool:
    if config is None:
        return False
    mode = getattr(config, "prompt_mode", "auto")
    if mode == "local":
        return True
    if mode == "cloud":
        return False
    return getattr(config, "provider", "openai") in _LOCAL_PROVIDERS


def load_formatted_prompt(prompt_file, project=None, config=None, **kwargs):
    """
    Load a prompt file with provider-aware local/cloud selection.

    Resolution order:
    1. project.prompts/<prompt_file>  (user override, always)
    2. prompts/local/<prompt_file>    (if local mode is active and file exists)
    3. prompts/<prompt_file>          (library default)

    Args:
        prompt_file (str): Filename of the prompt (e.g. "causal_classify.txt").
        project: ProjectContext with a 'prompts' attribute pointing to the user prompt dir.
        config: PipelineConfig; used to determine prompt_mode and provider.
        **kwargs: Template variables substituted into the prompt via str.format().

    Returns:
        str: The formatted prompt string.
    """
    use_local = _should_use_local_prompts(config)
    user_prompt_dir = Path(project.prompts) if project and hasattr(project, "prompts") else None

    user_path = user_prompt_dir / prompt_file if user_prompt_dir else None
    local_path = LOCAL_PROMPT_DIR / prompt_file
    default_path = DEFAULT_PROMPT_DIR / prompt_file

    if user_path and user_path.exists():
        prompt_text = user_path.read_text(encoding="utf-8")
    elif use_local and local_path.exists():
        prompt_text = local_path.read_text(encoding="utf-8")
    elif default_path.exists():
        prompt_text = default_path.read_text(encoding="utf-8")
    else:
        raise FileNotFoundError(
            f"Prompt '{prompt_file}' not found in user or default prompt directories."
        )

    if kwargs:
        prompt_text = prompt_text.format(**kwargs)

    return prompt_text
