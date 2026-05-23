import os
from huggingface_hub import InferenceClient

from config import OSS_MODEL_ID, OSS_MAX_NEW_TOKENS, OSS_TEMPERATURE
from models.base_client import BaseAssistantClient


class HFOSSClient(BaseAssistantClient):

    def __init__(self):
        api_key = os.getenv("HF_API_KEY")
        if not api_key:
            raise EnvironmentError("HF_API_KEY is not set in environment")
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=api_key,
        )
        self.model_id = OSS_MODEL_ID

    def chat(self, messages: list[dict]) -> dict:
        error = self._validate_messages(messages)
        if error:
            return error

        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=OSS_MAX_NEW_TOKENS,
                temperature=OSS_TEMPERATURE,
            )
            text = completion.choices[0].message.content
            return {"_code": 200, "_msg": "ok", "_data": text}

        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "429" in msg:
                return {"_code": 429, "_msg": f"HF rate limit: {e}", "_data": None}
            if "timeout" in msg or "503" in msg or "unavailable" in msg:
                return {"_code": 503, "_msg": f"HF model unavailable: {e}", "_data": None}
            return {"_code": 500, "_msg": f"HF inference error: {e}", "_data": None}
