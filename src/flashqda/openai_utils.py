import os
import openai
import time
from pathlib import Path

def get_openai_api_key(api_key_filename="openai_api_key.txt", project_root=None):
    """
    Initialize the OpenAI API key from a file.

    Args:
        api_key_filename (str): Name of the API key file (default: "openai_api_key.txt").
        project_root (str or Path, optional): Directory to look for the API key file.
            Defaults to current working directory.

    Raises:
        FileNotFoundError: If the API key file is not found.
        RuntimeError: If the API key file is empty.
    """
    base_path = Path(project_root) if project_root else Path.cwd()
    key_path = base_path / api_key_filename

    if key_path.exists():
        with open(key_path, "r") as f:
            key = f.read().strip()
        if not key:
            raise RuntimeError(f"OpenAI API key file at {key_path} is empty.")
        os.environ["OPENAI_API_KEY"] = key
        openai.api_key = key
    else:
        raise FileNotFoundError(f"OpenAI API key file not found at {key_path}")

import openai
import time
from openai import OpenAIError

def send_to_openai(
    system_prompt,
    user_prompt,
    model="gpt-4o",
    temperature=0.0,
    max_retries=3,
    sleep_seconds=5,
    timeout=15,
    response_format=None
):
    if response_format is None:
        response_format={"type": "json_object"}
        
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
                response_format=response_format,
            )
            content = response.choices[0].message["content"].strip()
            return content
        except OpenAIError as e:
            if attempt < max_retries - 1:
                time.sleep(sleep_seconds)
            else:
                raise RuntimeError(f"OpenAI API call failed after {max_retries} attempts: {e}")

