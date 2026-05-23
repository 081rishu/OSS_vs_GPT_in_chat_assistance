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
# Telemetry
# ---------------------------------------------------------------------------
TELEMETRY_FILE       = "telemetry_history.jsonl"

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_RESULTS_FILE    = "evaluation/eval_results.json"
EVAL_TEST_SUITE_FILE = "evaluation/test_suite.json"
EVAL_SCORE_MIN       = 1
EVAL_SCORE_MAX       = 5
