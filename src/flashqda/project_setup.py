# project_setup.py — handles initialization and validation of project directories

import os

def initialize_project(project_path):
    os.makedirs(os.path.join(project_path, "data"), exist_ok=True)
    os.makedirs(os.path.join(project_path, "results"), exist_ok=True)
    os.makedirs(os.path.join(project_path, "prompts"), exist_ok=True)
    print(f"Initialized project structure in: {project_path}")


def validate_project_structure(project_path):
    required_dirs = ["data", "results"]
    missing = [d for d in required_dirs if not os.path.isdir(os.path.join(project_path, d))]
    if missing:
        raise FileNotFoundError(f"Missing required directories: {', '.join(missing)}")
