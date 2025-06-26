from pathlib import Path

DEFAULT_PROMPT_DIR = Path(__file__).parent / "prompts"

def load_formatted_prompt(prompt_file, project=None, **kwargs):
    """
    Load prompt text, preferring user prompt directory in project.prompts,
    falling back to default prompt directory in library.

    Args:
        prompt_file (str): Name of the prompt file.
        project (optional): ProjectContext object with 'prompts' attribute.
        **kwargs: Formatting variables for the prompt template.

    Returns:
        str: The formatted prompt string.
    """
    user_prompt_dir = Path(project.prompts) if project and hasattr(project, "prompts") else None

    # Candidate paths
    user_path = user_prompt_dir / prompt_file if user_prompt_dir else None
    default_path = DEFAULT_PROMPT_DIR / prompt_file

    # Load from user prompt dir if exists
    if user_path and user_path.exists():
        prompt_text = user_path.read_text(encoding="utf-8")
    elif default_path.exists():
        prompt_text = default_path.read_text(encoding="utf-8")
    else:
        raise FileNotFoundError(f"Prompt '{prompt_file}' not found in user or default prompt directories.")

    # Format prompt with passed keyword args
    if kwargs:
        prompt_text = prompt_text.format(**kwargs)

    return prompt_text
