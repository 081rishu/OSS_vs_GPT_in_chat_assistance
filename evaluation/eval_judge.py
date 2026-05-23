import json
import os
import time
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# allow running as `python -m evaluation.eval_judge` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    JUDGE_MODEL_ID, JUDGE_MAX_TOKENS, JUDGE_TEMPERATURE,
    EVAL_TEST_SUITE_FILE, EVAL_RESULTS_FILE,
)
from prompts import JUDGE_SYSTEM_PROMPT, JUDGE_SCORE_PROMPT
from models.hf_oss_client import HFOSSClient
from models.openai_frontier_client import OpenAIFrontierClient
from core.memory import LazyMemoryManager
from core.guardrails import InputGuardrail, OutputAuditor
from prompts import ASSISTANT_SYSTEM_PROMPT


load_dotenv()


def _load_test_suite() -> dict:
    path = Path(EVAL_TEST_SUITE_FILE)
    if not path.exists():
        return {"_code": 404, "_msg": f"test suite not found: {path}", "_data": None}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {"_code": 200, "_msg": "ok", "_data": data}


def _run_bucket(client, memory: LazyMemoryManager, bucket: dict) -> dict:
    """Run all turns in a bucket and return list of (user_turn, response, latency_ms)."""
    guard_in = InputGuardrail()
    guard_out = OutputAuditor()
    results = []

    for turn in bucket["turns"]:
        user_text = turn["content"]

        check = guard_in.check(user_text)
        if check["_code"] != 200:
            results.append({
                "user": user_text,
                "response": check["_data"],
                "latency_ms": 0,
                "guard_status": check["_msg"],
            })
            memory.add_turn("user", user_text)
            memory.add_turn("assistant", check["_data"])
            continue

        memory.add_turn("user", user_text)
        msgs_result = memory.get_messages(ASSISTANT_SYSTEM_PROMPT)
        if msgs_result["_code"] != 200:
            return {"_code": 500, "_msg": msgs_result["_msg"], "_data": None}

        t0 = time.perf_counter()
        result = client.chat(msgs_result["_data"])
        latency_ms = (time.perf_counter() - t0) * 1000

        if result["_code"] != 200:
            response = f"[error {result['_code']}] {result['_msg']}"
            guard_status = f"error:{result['_code']}"
        else:
            audit = guard_out.check(result["_data"])
            response = audit["_data"]
            guard_status = "pass" if audit["_code"] == 200 else audit["_msg"]

        memory.add_turn("assistant", response)
        results.append({
            "user": user_text,
            "response": response,
            "latency_ms": round(latency_ms, 2),
            "guard_status": guard_status,
        })

    return {"_code": 200, "_msg": "ok", "_data": results}


def _judge_response(judge: OpenAI, bucket_name: str, criteria: str, history: str, response: str) -> dict:
    prompt = JUDGE_SCORE_PROMPT.format(
        bucket_name=bucket_name,
        criteria=criteria,
        history=history,
        response=response,
    )
    try:
        completion = judge.chat.completions.create(
            model=JUDGE_MODEL_ID,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=JUDGE_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content
        parsed = json.loads(raw)
        if "score" not in parsed or "reason" not in parsed:
            return {"_code": 500, "_msg": "judge returned incomplete JSON", "_data": None}
        return {"_code": 200, "_msg": "ok", "_data": parsed}
    except Exception as e:
        return {"_code": 500, "_msg": f"judge error: {e}", "_data": None}


def run_evaluation():
    suite_result = _load_test_suite()
    if suite_result["_code"] != 200:
        print(suite_result["_msg"])
        return

    suite = suite_result["_data"]
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set")
        return

    judge = OpenAI(api_key=api_key)
    oss_client = HFOSSClient()
    frontier_client = OpenAIFrontierClient()

    all_results = []

    for bucket in suite["buckets"]:
        print(f"\n--- Bucket: {bucket['name']} ---")
        bucket_result = {"bucket_id": bucket["id"], "bucket_name": bucket["name"], "turns": []}

        for model_label, client in [("oss", oss_client), ("frontier", frontier_client)]:
            memory = LazyMemoryManager()
            run = _run_bucket(client, memory, bucket)
            if run["_code"] != 200:
                print(f"  [{model_label}] failed: {run['_msg']}")
                continue

            turns = run["_data"]
            history_so_far = ""
            scored_turns = []

            for i, turn in enumerate(turns):
                history_so_far += f"User: {turn['user']}\nAssistant: {turn['response']}\n\n"
                judge_result = _judge_response(
                    judge,
                    bucket["name"],
                    bucket["criteria"],
                    history_so_far,
                    turn["response"],
                )
                score = judge_result["_data"]["score"] if judge_result["_code"] == 200 else None
                reason = judge_result["_data"]["reason"] if judge_result["_code"] == 200 else judge_result["_msg"]

                scored_turns.append({
                    "turn_index": i + 1,
                    "user": turn["user"],
                    "response": turn["response"],
                    "latency_ms": turn["latency_ms"],
                    "guard_status": turn["guard_status"],
                    "score": score,
                    "reason": reason,
                })
                print(f"  [{model_label}] turn {i+1} → score={score} | {reason}")

            avg_score = round(
                sum(t["score"] for t in scored_turns if t["score"] is not None) /
                max(1, sum(1 for t in scored_turns if t["score"] is not None)),
                2,
            )
            avg_latency = round(
                sum(t["latency_ms"] for t in scored_turns) / max(1, len(scored_turns)),
                2,
            )

            bucket_result["turns"].append({
                "model": model_label,
                "avg_score": avg_score,
                "avg_latency_ms": avg_latency,
                "details": scored_turns,
            })

        all_results.append(bucket_result)

    output_path = Path(EVAL_RESULTS_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    _print_summary(all_results)


def _print_summary(results: list):
    print("\n========== EVALUATION SUMMARY ==========")
    for bucket in results:
        print(f"\n{bucket['bucket_name']}")
        for model_result in bucket["turns"]:
            print(
                f"  {model_result['model']:10s}  avg_score={model_result['avg_score']}  "
                f"avg_latency={model_result['avg_latency_ms']}ms"
            )


if __name__ == "__main__":
    run_evaluation()
