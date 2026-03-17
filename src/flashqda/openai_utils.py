import os
import time
import openai
from openai import OpenAI
from openai import OpenAIError

import os
import time
from openai import OpenAI, OpenAIError

_client = None  # Global client placeholder

from pathlib import Path
import os
from openai import OpenAI, OpenAIError

_client = None  # Global client

def get_openai_api_key(api_key=None, api_key_filename="openai_api_key.txt", project_root=None):
    """
    Initialize the OpenAI API key and client.

    Priority:
    1. Use the `api_key` argument if provided.
    2. Else, try reading from the given `api_key_filename` in `project_root` or CWD.
    3. Else, use the OPENAI_API_KEY environment variable.

    Args:
        api_key (str, optional): Explicit API key.
        api_key_filename (str): Name of the file containing the API key.
        project_root (str or Path, optional): Directory to look for the API key file.

    Raises:
        FileNotFoundError: If file is specified but not found.
        RuntimeError: If the key file is empty.
        OpenAIError: If no valid API key is available.
    """
    global _client

    if api_key:
        key = api_key
    else:
        # Try file
        base_path = Path(project_root) if project_root else Path.cwd()
        key_path = base_path / api_key_filename

        if key_path.exists():
            with open(key_path, "r") as f:
                key = f.read().strip()
            if not key:
                raise RuntimeError(f"OpenAI API key file at {key_path} is empty.")
        else:
            # Fallback to env var
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise OpenAIError(
                    f"No API key provided, and file not found at {key_path}, "
                    "and OPENAI_API_KEY environment variable not set."
                )

    # Store in env for downstream compatibility
    os.environ["OPENAI_API_KEY"] = key
    _client = OpenAI(api_key=key)

def get_client():
    """
    Returns the OpenAI client if initialized; raises an error otherwise.
    """
    if _client is None:
        raise OpenAIError("OpenAI client not initialized. Call get_openai_api_key() first.")
    return _client

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
        response_format = {"type": "json_object"}

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(max_retries):
        try:
            client = get_client()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
                response_format=response_format,
            )
            content = response.choices[0].message.content.strip()
            return content
        except OpenAIError as e:
            if attempt < max_retries - 1:
                time.sleep(sleep_seconds)
            else:
                raise RuntimeError(f"OpenAI API call failed after {max_retries} attempts: {e}")
