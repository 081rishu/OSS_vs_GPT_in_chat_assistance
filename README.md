# AI Assistant Comparison — Ollive Assignment

A production-grade dual-assistant evaluation pipeline that runs **Qwen 2.5-0.5B** (open-source) and **GPT-4o-mini** (frontier) side-by-side in a Streamlit UI, with identical guardrail, memory, and telemetry pipelines wrapping both models.

---

## Quick Start

```bash
# 1. Clone and create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY and HF_API_KEY

# 4. Run the Streamlit app
streamlit run app.py

# 5. (Optional) Run the evaluation suite
python -m evaluation.eval_judge
```

---

## Project Structure

```
ollive_assignment/
├── app.py                        # Streamlit UI — two columns + metrics sidebar
├── config.py                     # All constants: model IDs, thresholds, file paths
├── prompts.py                    # All LLM prompt strings as named constants
├── .env.example                  # Credential template (copy to .env)
├── requirements.txt
│
├── core/
│   ├── memory.py                 # Lazy hybrid memory manager (Turn-6 compaction)
│   ├── guardrails.py             # 3-stage input/output safety pipeline
│   └── logger.py                 # Structured JSONL telemetry writer
│
├── models/
│   ├── base_client.py            # Abstract base class enforcing shared interface
│   ├── hf_oss_client.py          # Qwen 2.5-0.5B via HF Serverless Inference API
│   └── openai_frontier_client.py # GPT-4o-mini via OpenAI SDK
│
├── evaluation/
│   ├── test_suite.json           # 3-bucket, 11-turn evaluation dataset
│   ├── eval_judge.py             # GPT-4o LLM-as-a-Judge scoring loop
│   └── eval_results.json         # Output: per-turn scores, latency, guard status
│
└── architectural_decisions/      # Design briefs and tradeoff logs
```

---

## Architecture Decisions

### Model Selection

| Role | Model | Reason |
|---|---|---|
| OSS Assistant | `Qwen/Qwen2.5-0.5B-Instruct` | Free HF Serverless inference, deployable on HF Spaces, explicitly recommended in assignment |
| Frontier Assistant | `gpt-4o-mini` | Best cost-to-capability ratio in OpenAI's lineup ($0.15/1M input) |
| Eval Judge | `gpt-4o` | More capable than either tested model — avoids self-grading bias |

> **Note on Qwen 2.5-0.5B:** At 490M parameters this is a deliberately small model. It will exhibit weaker multi-turn coherence and adversarial robustness than GPT-4o-mini. This is an intentional tradeoff — free inference and zero infrastructure cost vs. raw capability. The evaluation results quantify this gap.

### Memory — Lazy Hybrid (Turn-6 Compaction)

Each assistant maintains two independent state primitives:

- `recent_history` — a FIFO sliding window of raw chat turns
- `metadata_profile` — a persistent key-value dict of extracted user facts

**Lifecycle:**
- **Turns 1–5:** Prompts flow directly into the sliding window. Zero extraction overhead, minimum TTFT.
- **Turn 6:** A single out-of-band extraction call fires over the existing history, identifying permanent variables (name, language, constraints) and writing them to `metadata_profile`.
- **Turn 7+:** Oldest raw turns are pruned. The `metadata_profile` is injected into the system prompt on every turn, so pinned facts are never lost.

**Why not eager summarization?** Summarizing on every turn doubles TTFT on trigger turns and creates a "telephone game" effect where cascading summaries erode specific details.

**Why not vector RAG?** For 20–50 turn evaluations, a vector database adds dependency complexity and cold-start risk without meaningful recall benefit.

### Guardrails — 3-Stage Decoupled Pipeline

Every prompt passes through three independent safety stages before and after model inference:

**Stage 1 — Input Filter** (pre-inference)
- Tier 1: Regex matching against known jailbreak phrases (~0–5ms)
- Tier 2: OpenAI Moderation API for semantic hazard classification (~100–300ms)
- If either fires: immediate refusal, model inference is skipped entirely
- Combined worst-case early-exit: sub-350ms

**Stage 2 — System Alignment** (in-payload)
- An immutable system prompt is injected at position 0 of every message array
- Anchors personality, enforces refusal behavior, prohibits subjective political/religious opinions

**Stage 3 — Output Auditor** (post-inference)
- Scans for system prompt leakage tokens
- Detects repetition loops (common failure mode in small OSS models under adversarial stress)
- Replaces flagged output with a clean fallback before rendering to the UI

### Evaluation — LLM-as-a-Judge

Three multi-turn conversation buckets designed to expose failure modes that standard benchmarks miss:

| Bucket | Turns | Tests |
|---|---|---|
| Factual Retention & Amnesia | 6 | Context carry, API hallucination, versioned library facts |
| Adversarial & Jailbreak | 3 | Regex bypass attempts, roleplay override, system prompt extraction |
| Bias & Compliance | 2 | Discriminatory premise traps, neutrality under pressure |

`gpt-4o` scores each response 1–5 using a strict rubric with all five levels defined. Results are written to `evaluation/eval_results.json` with per-turn scores, latency, and guard status for both models.

---

## Tradeoffs

| Decision | Tradeoff |
|---|---|
| Qwen 2.5-0.5B as OSS model | Free and deployable, but significantly weaker than larger OSS alternatives. A 7B+ model would perform closer to frontier but requires GPU infrastructure. |
| HF Serverless Inference | Zero cost, but the free tier has cold-start delays (10–30s on first request) and rate throttling under sustained load. |
| Flat JSON for telemetry and eval results | Simple, portable, zero dependencies. Not suitable for concurrent writes or large-scale runs. |
| OpenAI Moderation API for Stage 1 Tier 2 | Free and no cold-start, but introduces an OpenAI dependency into the otherwise OSS pipeline path. |
| Turn-6 compaction threshold | Balances TTFT optimization against memory retention. A lower threshold (e.g., Turn 3) retains more but adds latency on shorter conversations. |
| Character-based token estimation | Used when the HF Serverless API does not return usage metadata. Approximation only — actual token counts may differ by ±15%. |

---

## What I Would Improve With More Time

1. **Larger OSS model** — Qwen 2.5-7B or Llama 3.2-3B would close much of the capability gap with GPT-4o-mini while still being deployable on free-tier GPU instances (e.g., HF Spaces ZeroGPU).

2. **In-memory vector search for memory** — Replace the flat `metadata_profile` dict with a local embedding-based retrieval layer (manual dot-product over OpenAI embeddings). No external DB needed, but enables semantic fact lookup rather than just key-value storage.

3. **Streaming responses** — Both the HF Serverless API and OpenAI SDK support token streaming. Wiring this into Streamlit (`st.write_stream`) would dramatically improve perceived latency for the user.

4. **Richer evaluation dataset** — Expand from 3 buckets to 6–8, covering tool use, multi-language code generation, and long-horizon multi-session memory recall.

5. **Persistent sessions** — Currently memory resets on page reload. A session ID + lightweight SQLite store would enable returning users to resume conversations.

6. **Real token counting** — Replace character-based estimation with `tiktoken` for OpenAI models and the HF tokenizer for Qwen, giving accurate cost reporting in the telemetry log.

---

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Used for GPT-4o-mini (frontier assistant), memory extraction (Turn-6 compaction), OpenAI Moderation API (Stage 1 guardrail), and GPT-4o eval judge |
| `HF_API_KEY` | Hugging Face read token for Serverless Inference API (Qwen 2.5-0.5B) |

---

## Running the Evaluation

```bash
# From the repo root, with venv active and .env populated
python -m evaluation.eval_judge
```

Results are printed to stdout as a summary table and written in full to `evaluation/eval_results.json`.
