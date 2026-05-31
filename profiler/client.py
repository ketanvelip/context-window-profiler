import os
import time
from together import Together


def get_client() -> Together:
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise EnvironmentError("TOGETHER_API_KEY environment variable is not set.")
    return Together(api_key=api_key)


def complete(client: Together, model: str, system: str, user: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=1024,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return ""
