import json
import os
from openai import OpenAI

from config import MEMORY_COMPACTION_TURN, MEMORY_MAX_HISTORY_TURNS
from prompts import MEMORY_EXTRACTION_PROMPT


class LazyMemoryManager:

    def __init__(self):
        self.recent_history: list[dict] = []
        self.metadata_profile: dict = {}
        self._turn_count: int = 0
        self._compaction_done: bool = False

    def add_turn(self, role: str, content: str) -> dict:
        if role not in ("user", "assistant"):
            return {"_code": 400, "_msg": f"Invalid role '{role}'", "_data": None}
        if not content or not isinstance(content, str):
            return {"_code": 400, "_msg": "content must be a non-empty string", "_data": None}

        self.recent_history.append({"role": role, "content": content})

        if role == "user":
            self._turn_count += 1

        if self._turn_count >= MEMORY_COMPACTION_TURN and not self._compaction_done:
            self._run_compaction()

        return {"_code": 200, "_msg": "ok", "_data": None}

    def get_messages(self, system_prompt: str) -> dict:
        if not system_prompt or not isinstance(system_prompt, str):
            return {"_code": 400, "_msg": "system_prompt must be a non-empty string", "_data": None}

        system_content = system_prompt
        if self.metadata_profile:
            profile_str = json.dumps(self.metadata_profile, indent=2)
            system_content += f"\n\nKnown facts about the user:\n{profile_str}"

        messages = [{"role": "system", "content": system_content}] + self.recent_history
        return {"_code": 200, "_msg": "ok", "_data": messages}

    def reset(self) -> dict:
        self.recent_history = []
        self.metadata_profile = {}
        self._turn_count = 0
        self._compaction_done = False
        return {"_code": 200, "_msg": "ok", "_data": None}

    def _run_compaction(self):
        history_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in self.recent_history
        )
        prompt = MEMORY_EXTRACTION_PROMPT.format(history=history_text)

        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=256,
                )
                raw = response.choices[0].message.content.strip()
                extracted = json.loads(raw)
                if isinstance(extracted, dict):
                    self.metadata_profile.update(extracted)
        except Exception:
            pass

        # prune oldest turns, keep the window bounded
        if len(self.recent_history) > MEMORY_MAX_HISTORY_TURNS:
            self.recent_history = self.recent_history[-MEMORY_MAX_HISTORY_TURNS:]

        self._compaction_done = True
