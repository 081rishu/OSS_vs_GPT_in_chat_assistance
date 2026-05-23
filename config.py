import os

# ---------------------------------------------------------------------------
# OSS Model
# ---------------------------------------------------------------------------
OSS_MODEL_ID         = "Qwen/Qwen2.5-0.5B-Instruct"
OSS_MAX_NEW_TOKENS   = 512
OSS_TEMPERATURE      = 0.7

# ---------------------------------------------------------------------------
# Frontier Model
# ---------------------------------------------------------------------------
FRONTIER_MODEL_ID    = "gpt-4o-mini"
FRONTIER_MAX_TOKENS  = 1024
FRONTIER_TEMPERATURE = 0.7

# ---------------------------------------------------------------------------
# Judge Model (eval only)
# ---------------------------------------------------------------------------
JUDGE_MODEL_ID       = "gpt-4o"
JUDGE_MAX_TOKENS     = 1024
JUDGE_TEMPERATURE    = 0.0

# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
MEMORY_COMPACTION_TURN   = 6     # turn at which lazy extraction fires
MEMORY_MAX_HISTORY_TURNS = 10    # max raw turns kept after compaction

# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
MODERATION_ENDPOINT  = "https://api.openai.com/v1/moderations"
GUARDRAIL_CATEGORIES = [
    "hate", "hate/threatening", "harassment", "harassment/threatening",
    "self-harm", "self-harm/intent", "self-harm/instructions",
    "sexual", "sexual/minors",
    "violence", "violence/graphic",
    "illicit", "illicit/violent",
]

# ---------------------------------------------------------------------------
# Demo session limits (public HF Spaces deployment protection)
# ---------------------------------------------------------------------------
DEMO_MAX_TURNS       = 20      # max turns per session before throttling
DEMO_MAX_TOKENS      = 8000    # approximate total token budget per session
DEMO_CONTACT_EMAIL   = "freeloadermr@gmail.com"
DEMO_PASSKEY         = os.getenv("DEMO_PASSKEY", "")  # set in .env — resets usage on hard stop

# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------
TELEMETRY_FILE       = "telemetry_history.jsonl"
TELEMETRY_ENABLED    = os.getenv("TELEMETRY_ENABLED", "true").lower() == "true"

# ---------------------------------------------------------------------------
# OSS inference provider — override via env var for fallback deployments
# HF Spaces may need "nebius" or "together" if hf-inference blocks the model
# ---------------------------------------------------------------------------
OSS_PROVIDER         = os.getenv("OSS_PROVIDER", "hf-inference")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_RESULTS_FILE    = "evaluation/eval_results.json"
EVAL_TEST_SUITE_FILE = "evaluation/test_suite.json"
EVAL_SCORE_MIN       = 1
EVAL_SCORE_MAX       = 5
