from abc import ABC, abstractmethod


class BaseAssistantClient(ABC):

    @abstractmethod
    def chat(self, messages: list[dict]) -> dict:
        """
        Send a compiled message array to the model and return a standard response.

        Args:
            messages: List of {"role": str, "content": str} dicts.
                      Must include the system prompt at index 0.

        Returns:
            {"_code": int, "_msg": str, "_data": str | None}
            _data is the raw response text on success, None on failure.
        """

    def _validate_messages(self, messages: list[dict]) -> dict | None:
        """
        Shared validation for the messages payload.
        Returns an error dict if invalid, None if valid.
        """
        if not messages or not isinstance(messages, list):
            return {"_code": 400, "_msg": "messages must be a non-empty list", "_data": None}
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                return {"_code": 400, "_msg": f"messages[{i}] is not a dict", "_data": None}
            if "role" not in msg or "content" not in msg:
                return {"_code": 400, "_msg": f"messages[{i}] missing 'role' or 'content'", "_data": None}
            if msg["role"] not in ("system", "user", "assistant"):
                return {"_code": 400, "_msg": f"messages[{i}] has invalid role '{msg['role']}'", "_data": None}
        return None
