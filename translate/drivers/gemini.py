import requests

from translate import TranslationError
from translate.prompt import SYSTEM_PROMPT


def translate(text: str, target_lang: str, model: str, endpoint: str, api_key: str) -> str:
    """Send full SRT text to Google Gemini generateContent API, return translated SRT text."""
    prompt = SYSTEM_PROMPT.format(target_lang=target_lang)
    url = f"{endpoint}/{model}:generateContent?key={api_key}"
    payload = {
        "system_instruction": {
            "parts": [{"text": prompt}],
        },
        "contents": [
            {"parts": [{"text": text}]},
        ],
        "generationConfig": {
            "temperature": 0,
        },
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)
    except requests.RequestException as exc:
        raise TranslationError(f"Request failed: {exc}") from exc

    if not response.ok:
        raise TranslationError(f"API error {response.status_code}: {response.text[:300]}")

    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise TranslationError(f"Unexpected response structure: {str(data)[:300]}") from exc
