import os
from openai import OpenAI

from config import FRONTIER_MODEL_ID, FRONTIER_MAX_TOKENS, FRONTIER_TEMPERATURE
from models.base_client import BaseAssistantClient


class OpenAIFrontierClient(BaseAssistantClient):

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set in environment")
        self.client = OpenAI(api_key=api_key)
        self.model_id = FRONTIER_MODEL_ID

    def chat(self, messages: list[dict]) -> dict:
        error = self._validate_messages(messages)
        if error:
            return error

        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=FRONTIER_MAX_TOKENS,
                temperature=FRONTIER_TEMPERATURE,
            )
            text = completion.choices[0].message.content
            return {"_code": 200, "_msg": "ok", "_data": text}

        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "429" in msg:
                return {"_code": 429, "_msg": f"OpenAI rate limit: {e}", "_data": None}
            if "timeout" in msg or "503" in msg or "unavailable" in msg:
                return {"_code": 503, "_msg": f"OpenAI unavailable: {e}", "_data": None}
            return {"_code": 500, "_msg": f"OpenAI error: {e}", "_data": None}
