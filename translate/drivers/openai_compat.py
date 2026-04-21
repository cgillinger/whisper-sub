import requests

from translate import TranslationError
from translate.prompt import SYSTEM_PROMPT


def translate(text: str, target_lang: str, model: str, endpoint: str, api_key: str) -> str:
    """Send full SRT text to an OpenAI-compatible API, return translated SRT text."""
    prompt = SYSTEM_PROMPT.format(target_lang=target_lang)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=300)
    except requests.RequestException as exc:
        raise TranslationError(f"Request failed: {exc}") from exc

    if not response.ok:
        raise TranslationError(f"API error {response.status_code}: {response.text[:300]}")

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise TranslationError(f"Unexpected response structure: {str(data)[:300]}") from exc
