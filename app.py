import time
import streamlit as st
from dotenv import load_dotenv

from config import OSS_MODEL_ID, FRONTIER_MODEL_ID
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


_init_state()


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
        st.session_state.metrics = {
            "oss": {"latency_ms": None, "input_tokens": None, "output_tokens": None, "guard": "—"},
            "frontier": {"latency_ms": None, "input_tokens": None, "output_tokens": None, "guard": "—"},
        }
        st.rerun()


# ---------------------------------------------------------------------------
# Main chat columns
# ---------------------------------------------------------------------------
col_oss, col_frontier = st.columns(2)

with col_oss:
    st.subheader("Qwen 2.5-0.5B (OSS)")
    oss_container = st.container(height=520)

with col_frontier:
    st.subheader("GPT-4o-mini (Frontier)")
    frontier_container = st.container(height=520)

# render existing history
for entry in st.session_state.chat_log:
    with oss_container:
        with st.chat_message(entry["role"]):
            st.markdown(entry["oss_content"])
    with frontier_container:
        with st.chat_message(entry["role"]):
            st.markdown(entry["frontier_content"])

# chat input
user_input = st.chat_input("Type a message...")

if user_input:
    # show user message immediately in both columns
    with oss_container:
        with st.chat_message("user"):
            st.markdown(user_input)
    with frontier_container:
        with st.chat_message("user"):
            st.markdown(user_input)

    # run both turns
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

    # persist to chat log and update metrics
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
