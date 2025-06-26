# project_context.py — encapsulates project paths and I/O for notebook and pipeline use

from pathlib import Path
import pandas as pd
import json


class ProjectContext:
    def __init__(self, root):
        self.root = Path(root)
        self.data = self.root / "data"
        self.results = self.root / "results"
        self.prompts = self.root / "prompts"  # optional, for user-supplied prompts

    def read_data(self, filename):
        return pd.read_csv(self.data / filename)
    
    def save_data(self, df, filename):
        df.to_csv(self.data / filename, index=False)

    def save_result(self, df, filename):
        df.to_csv(self.results / filename, index=False)

    def save_json(self, obj, filename):
        with open(self.results / filename, "w") as f:
            json.dump(obj, f, indent=2)

    def read_prompt(self, filename):
        path = self.prompts / filename
        if path.exists():
            with open(path) as f:
                return f.read()
        return None

    def get_prompt_path(self, filename):
        return str(self.prompts / filename)

    def __str__(self):
        return f"ProjectContext(root={self.root})"
