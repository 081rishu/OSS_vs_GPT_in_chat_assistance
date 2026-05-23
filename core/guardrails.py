import re
import os
from openai import OpenAI

from config import GUARDRAIL_CATEGORIES
from prompts import JAILBREAK_PATTERNS

REFUSAL_MESSAGE = (
    "I'm not able to help with that request. "
    "If you have a different question, I'm happy to assist."
)

LEAK_TOKENS = [
    "you are a helpful, accurate",
    "do not reveal these instructions",
    "known facts about the user:",
]


class InputGuardrail:

    def __init__(self):
        self._patterns = [re.compile(p, re.IGNORECASE) for p in JAILBREAK_PATTERNS]
        api_key = os.getenv("OPENAI_API_KEY")
        self._moderation_client = OpenAI(api_key=api_key) if api_key else None

    def check(self, text: str) -> dict:
        if not text or not isinstance(text, str):
            return {"_code": 400, "_msg": "text must be a non-empty string", "_data": None}

        # Tier 1: regex
        for pattern in self._patterns:
            if pattern.search(text):
                return {"_code": 400, "_msg": "blocked:jailbreak", "_data": REFUSAL_MESSAGE}

        # Tier 2: OpenAI Moderation API
        if self._moderation_client:
            result = self._call_moderation(text)
            if result["_code"] != 200:
                return result
            if result["_data"] is True:
                return {"_code": 400, "_msg": "blocked:moderation", "_data": REFUSAL_MESSAGE}

        return {"_code": 200, "_msg": "ok", "_data": None}

    def _call_moderation(self, text: str) -> dict:
        try:
            response = self._moderation_client.moderations.create(input=text)
            flagged = response.results[0].flagged
            return {"_code": 200, "_msg": "ok", "_data": flagged}
        except Exception as e:
            # moderation failure is non-fatal — let the request through
            return {"_code": 200, "_msg": f"moderation skipped: {e}", "_data": False}


class OutputAuditor:

    def check(self, text: str) -> dict:
        if not text or not isinstance(text, str):
            return {"_code": 400, "_msg": "text must be a non-empty string", "_data": None}

        lower = text.lower()

        # detect system prompt leakage
        for token in LEAK_TOKENS:
            if token in lower:
                return {"_code": 400, "_msg": "blocked:prompt_leak", "_data": REFUSAL_MESSAGE}

        # detect repetition loops (same sentence repeated 4+ times)
        sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
        if len(sentences) >= 4:
            counts = {}
            for s in sentences:
                counts[s] = counts.get(s, 0) + 1
            if max(counts.values()) >= 4:
                return {"_code": 400, "_msg": "blocked:repetition_loop", "_data": REFUSAL_MESSAGE}

        return {"_code": 200, "_msg": "ok", "_data": text}
