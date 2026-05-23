import json
from datetime import datetime, timezone

from config import TELEMETRY_FILE


class TelemetryLogger:

    def log(
        self,
        model_variant: str,
        turn_count: int,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        guardrail_status: str,
    ) -> dict:
        if not model_variant or not isinstance(model_variant, str):
            return {"_code": 400, "_msg": "model_variant must be a non-empty string", "_data": None}
        if not isinstance(turn_count, int) or turn_count < 0:
            return {"_code": 400, "_msg": "turn_count must be a non-negative integer", "_data": None}
        if not isinstance(latency_ms, (int, float)) or latency_ms < 0:
            return {"_code": 400, "_msg": "latency_ms must be a non-negative number", "_data": None}

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_variant": model_variant,
            "turn_count": turn_count,
            "latency_ms": round(latency_ms, 2),
            "tokens_processed": {
                "input": input_tokens,
                "output": output_tokens,
            },
            "guardrail_status": guardrail_status,
        }

        try:
            with open(TELEMETRY_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            return {"_code": 200, "_msg": "ok", "_data": entry}
        except Exception as e:
            return {"_code": 500, "_msg": f"failed to write telemetry: {e}", "_data": None}
