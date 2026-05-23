import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from config import OSS_MODEL_ID, FRONTIER_MODEL_ID, DEMO_MAX_TURNS, DEMO_MAX_TOKENS, DEMO_CONTACT_EMAIL, DEMO_PASSKEY
from config import EVAL_RESULTS_FILE
from prompts import ASSISTANT_SYSTEM_PROMPT
from models.hf_oss_client import HFOSSClient
from models.openai_frontier_client import OpenAIFrontierClient
from core.memory import LazyMemoryManager
from core.guardrails import InputGuardrail, OutputAuditor
from core.logger import TelemetryLogger

load_dotenv()

st.set_page_config(page_title="AI Assistant Comparison", layout="wide")
st.title("AI Assistant Comparison")
st.caption(f"**OSS:** {OSS_MODEL_ID}  |  **Frontier:** {FRONTIER_MODEL_ID}")


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
def _init_state():
    if "oss_memory" not in st.session_state:
        st.session_state.oss_memory = LazyMemoryManager()
    if "frontier_memory" not in st.session_state:
        st.session_state.frontier_memory = LazyMemoryManager()
    if "oss_client" not in st.session_state:
        st.session_state.oss_client = HFOSSClient()
    if "frontier_client" not in st.session_state:
        st.session_state.frontier_client = OpenAIFrontierClient()
    if "guardrail_in" not in st.session_state:
        st.session_state.guardrail_in = InputGuardrail()
    if "guardrail_out" not in st.session_state:
        st.session_state.guardrail_out = OutputAuditor()
    if "logger" not in st.session_state:
        st.session_state.logger = TelemetryLogger()
    if "chat_log" not in st.session_state:
        # list of {role, oss_content, frontier_content}
        st.session_state.chat_log = []
    if "metrics" not in st.session_state:
        # last turn metrics for sidebar display
        st.session_state.metrics = {
            "oss": {"latency_ms": None, "input_tokens": None, "output_tokens": None, "guard": "—"},
            "frontier": {"latency_ms": None, "input_tokens": None, "output_tokens": None, "guard": "—"},
        }
    if "session_turns" not in st.session_state:
        st.session_state.session_turns = 0
    if "session_tokens" not in st.session_state:
        st.session_state.session_tokens = 0


_init_state()


# ---------------------------------------------------------------------------
# Demo limit check
# ---------------------------------------------------------------------------
def _demo_limit_reached() -> str | None:
    """Returns a throttle message if the session has hit its limit, None otherwise."""
    if st.session_state.session_turns >= DEMO_MAX_TURNS:
        return (
            f"This demo is limited to **{DEMO_MAX_TURNS} turns** per session. "
            f"To continue or request extended access, contact "
            f"[{DEMO_CONTACT_EMAIL}](mailto:{DEMO_CONTACT_EMAIL})."
        )
    if st.session_state.session_tokens >= DEMO_MAX_TOKENS:
        return (
            f"This demo has reached its token budget for this session. "
            f"To continue or request extended access, contact "
            f"[{DEMO_CONTACT_EMAIL}](mailto:{DEMO_CONTACT_EMAIL})."
        )
    return None


# ---------------------------------------------------------------------------
# Core turn execution
# ---------------------------------------------------------------------------
def _run_turn(user_input: str, client, memory: LazyMemoryManager, model_label: str) -> tuple[str, dict]:
    """
    Runs a single assistant turn through the full pipeline.
    Returns (response_text, metrics_dict).
    """
    guard_in = st.session_state.guardrail_in
    guard_out = st.session_state.guardrail_out
    logger = st.session_state.logger

    # Stage 1: input guardrail
    check = guard_in.check(user_input)
    if check["_code"] != 200:
        metrics = {"latency_ms": 0, "input_tokens": 0, "output_tokens": 0, "guard": check["_msg"]}
        logger.log(model_label, memory._turn_count, 0, 0, 0, check["_msg"])
        return check["_data"], metrics

    # add user turn to memory
    memory.add_turn("user", user_input)

    # build message payload
    msgs_result = memory.get_messages(ASSISTANT_SYSTEM_PROMPT)
    if msgs_result["_code"] != 200:
        return msgs_result["_msg"], {"latency_ms": 0, "input_tokens": 0, "output_tokens": 0, "guard": "error"}

    messages = msgs_result["_data"]

    # Stage 2 (system prompt alignment) is embedded in messages[0] via ASSISTANT_SYSTEM_PROMPT

    # call model
    t0 = time.perf_counter()
    result = client.chat(messages)
    latency_ms = (time.perf_counter() - t0) * 1000

    if result["_code"] != 200:
        metrics = {"latency_ms": round(latency_ms), "input_tokens": 0, "output_tokens": 0, "guard": f"error:{result['_code']}"}
        logger.log(model_label, memory._turn_count, latency_ms, 0, 0, f"error:{result['_code']}")
        # surface billing errors as a friendly contact message
        if result["_code"] in (429, 503) and "billing" in result["_msg"].lower() or "quota" in result["_msg"].lower():
            return (
                f"This demo has reached its usage limit. "
                f"To continue, contact [{DEMO_CONTACT_EMAIL}](mailto:{DEMO_CONTACT_EMAIL})."
            ), metrics
        return f"⚠️ {result['_msg']}", metrics

    raw_response = result["_data"]

    # Stage 3: output auditor
    audit = guard_out.check(raw_response)
    guard_status = "pass" if audit["_code"] == 200 else audit["_msg"]
    final_response = audit["_data"] if audit["_code"] == 200 else audit["_data"]

    # estimate tokens (character-based approximation when SDK doesn't return usage)
    input_tokens = sum(len(m["content"]) // 4 for m in messages)
    output_tokens = len(raw_response) // 4

    # add assistant turn to memory
    memory.add_turn("assistant", final_response)

    # accumulate session token usage (only count frontier model to avoid double-counting)
    if model_label == "gpt-4o-mini":
        st.session_state.session_tokens += input_tokens + output_tokens

    metrics = {
        "latency_ms": round(latency_ms),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "guard": guard_status,
    }
    logger.log(model_label, memory._turn_count, latency_ms, input_tokens, output_tokens, guard_status)
    return final_response, metrics


# ---------------------------------------------------------------------------
# Sidebar — live metrics
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Live Metrics")
    turns_left = DEMO_MAX_TURNS - st.session_state.session_turns
    tokens_left = DEMO_MAX_TOKENS - st.session_state.session_tokens
    st.caption(f"Demo quota: **{turns_left} turns** and **{tokens_left} tokens** remaining")
    st.divider()
    for label, key in [("Qwen 2.5-0.5B (OSS)", "oss"), ("GPT-4o-mini (Frontier)", "frontier")]:
        st.subheader(label)
        m = st.session_state.metrics[key]
        st.metric("Latency", f"{m['latency_ms']} ms" if m["latency_ms"] is not None else "—")
        st.metric("Input tokens", m["input_tokens"] if m["input_tokens"] is not None else "—")
        st.metric("Output tokens", m["output_tokens"] if m["output_tokens"] is not None else "—")
        guard_color = "🟢" if m["guard"] in ("—", "pass") else "🔴"
        st.markdown(f"**Guardrail:** {guard_color} `{m['guard']}`")
        st.divider()

    if st.button("Reset conversation"):
        st.session_state.oss_memory.reset()
        st.session_state.frontier_memory.reset()
        st.session_state.chat_log = []
        st.session_state.session_turns = 0
        st.session_state.session_tokens = 0
        st.session_state.metrics = {
            "oss": {"latency_ms": None, "input_tokens": None, "output_tokens": None, "guard": "—"},
            "frontier": {"latency_ms": None, "input_tokens": None, "output_tokens": None, "guard": "—"},
        }
        st.rerun()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_chat, tab_eval = st.tabs(["Chat", "Evaluation"])


# ---------------------------------------------------------------------------
# Tab 1: Chat
# ---------------------------------------------------------------------------
with tab_chat:
    col_oss, col_frontier = st.columns(2)

    with col_oss:
        st.subheader("Qwen 2.5-0.5B (OSS)")
        oss_container = st.container(height=520)

    with col_frontier:
        st.subheader("GPT-4o-mini (Frontier)")
        frontier_container = st.container(height=520)

    for entry in st.session_state.chat_log:
        with oss_container:
            with st.chat_message(entry["role"]):
                st.markdown(entry["oss_content"])
        with frontier_container:
            with st.chat_message(entry["role"]):
                st.markdown(entry["frontier_content"])

    user_input = st.chat_input("Type a message...")

    if user_input:
        limit_msg = _demo_limit_reached()
        if limit_msg:
            st.warning(limit_msg)
            passkey = st.text_input("Enter passkey to reset usage limits:", type="password", key="passkey_input")
            if passkey:
                if passkey == DEMO_PASSKEY:
                    st.session_state.session_turns = 0
                    st.session_state.session_tokens = 0
                    st.success("Usage reset. Continue chatting.")
                    st.rerun()
                else:
                    st.error("Invalid passkey.")
            st.stop()

        st.session_state.session_turns += 1

        with oss_container:
            with st.chat_message("user"):
                st.markdown(user_input)
        with frontier_container:
            with st.chat_message("user"):
                st.markdown(user_input)

        with oss_container:
            with st.chat_message("assistant"):
                with st.spinner(""):
                    oss_response, oss_metrics = _run_turn(
                        user_input,
                        st.session_state.oss_client,
                        st.session_state.oss_memory,
                        "qwen-0.5b",
                    )
                st.markdown(oss_response)

        with frontier_container:
            with st.chat_message("assistant"):
                with st.spinner(""):
                    frontier_response, frontier_metrics = _run_turn(
                        user_input,
                        st.session_state.frontier_client,
                        st.session_state.frontier_memory,
                        "gpt-4o-mini",
                    )
                st.markdown(frontier_response)

        st.session_state.chat_log.append({
            "role": "user",
            "oss_content": user_input,
            "frontier_content": user_input,
        })
        st.session_state.chat_log.append({
            "role": "assistant",
            "oss_content": oss_response,
            "frontier_content": frontier_response,
        })
        st.session_state.metrics["oss"] = oss_metrics
        st.session_state.metrics["frontier"] = frontier_metrics
        st.rerun()


# ---------------------------------------------------------------------------
# Tab 2: Evaluation
# ---------------------------------------------------------------------------
with tab_eval:
    st.subheader("Automated Evaluation Suite")
    st.caption(
        "Runs both assistants through the 5-bucket test suite and scores each response "
        "using GPT-4o as an impartial judge. Results are saved to `evaluation/eval_results.json`."
    )

    col_run, col_viz = st.columns(2)

    with col_run:
        if st.button("Run Evaluation", type="primary", use_container_width=True):
            with st.spinner("Running evaluation — this may take a few minutes..."):
                try:
                    from evaluation.eval_judge import run_evaluation
                    run_evaluation()
                    st.success("Evaluation complete. Results saved to `evaluation/eval_results.json`.")
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

    with col_viz:
        results_exist = Path(EVAL_RESULTS_FILE).exists()
        if st.button(
            "Show Visualizations",
            type="secondary",
            use_container_width=True,
            disabled=not results_exist,
            help="Run the evaluation first to generate results." if not results_exist else None,
        ):
            st.session_state.show_charts = True

    if not results_exist:
        st.info("No results yet. Click **Run Evaluation** to generate them.")

    if st.session_state.get("show_charts") and results_exist:
        with st.spinner("Generating charts..."):
            try:
                from evaluation.visualize import _load_results, _extract
                from evaluation.visualize import plot_scores, plot_latency, plot_safety_pass_rate, plot_cost_latency_table
                from evaluation.visualize import CHARTS_DIR

                CHARTS_DIR.mkdir(parents=True, exist_ok=True)
                results = _load_results()
                data    = _extract(results)

                plot_scores(data)
                plot_latency(data)
                plot_safety_pass_rate(data)
                plot_cost_latency_table(data)
            except Exception as e:
                st.error(f"Visualization failed: {e}")
                st.stop()

        st.markdown("---")

        chart_files = [
            ("evaluation/charts/scores_by_bucket.png",    "Evaluation Scores by Bucket"),
            ("evaluation/charts/safety_pass_rate.png",    "Safety & Quality Pass Rate"),
            ("evaluation/charts/latency_by_bucket.png",   "Response Latency by Bucket"),
            ("evaluation/charts/cost_latency_table.png",  "Cost & Latency Summary"),
        ]

        for path, title in chart_files:
            if Path(path).exists():
                st.markdown(f"**{title}**")
                st.image(path, use_container_width=True)
                st.divider()
