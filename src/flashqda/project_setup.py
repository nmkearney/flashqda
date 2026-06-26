# project_setup.py — handles initialization and validation of project directories

import os

def initialize_project(project_path):
    """
    Create the standard FlashQDA project directory structure.

    Creates three subdirectories under `project_path` if they do not already
    exist: `data/` (input documents), `analyses/` (pipeline outputs), and
    `prompts/` (optional user-supplied prompt overrides).

    Args:
        project_path (str or Path): Root directory for the project. Will be
            created if it does not exist.

    Returns:
        None
    """
    os.makedirs(os.path.join(project_path, "data"), exist_ok=True)
    os.makedirs(os.path.join(project_path, "analyses"), exist_ok=True)
    os.makedirs(os.path.join(project_path, "prompts"), exist_ok=True)
    print(f"Initialized project structure in: {project_path}")


def validate_project_structure(project_path):
    required_dirs = ["data"]
    missing = [d for d in required_dirs if not os.path.isdir(os.path.join(project_path, d))]
    if missing:
        raise FileNotFoundError(f"Missing required directories: {', '.join(missing)}")
