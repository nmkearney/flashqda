import os
import time
import openai
from openai import OpenAI
from openai import OpenAIError

client = OpenAI()  # Uses the key from env var OPENAI_API_KEY

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
