import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

from app.graph.builder import build_research_graph
from app.llm.client import llm_client
from app.llm.fake_provider import FakeProvider
from app.persistence.checkpointer import RedisCheckpointSaver
from app.persistence.redis_store import redis_store


def run_worker_process(run_id: str, topic: str, delay_per_step: float = 0.5):
    # Configure deterministic fake LLM with slight delay
    fake = FakeProvider()
    fake.artificial_delay_s = delay_per_step

    # Hook generate_structured to record every LLM call immediately to Redis
    orig_generate = fake.generate_structured
    client = redis_store.get_client()

    def recorded_generate(messages, output_schema, **kwargs):
        last_content = messages[-1].get("content", "")[:30] if messages else ""
        call_id = f"{output_schema.__name__}:{last_content}"
        client.rpush(f"chaos:raw_calls:{run_id}", call_id)
        return orig_generate(messages, output_schema, **kwargs)

    fake.generate_structured = recorded_generate
    llm_client.set_override_provider(fake)

    checkpointer = RedisCheckpointSaver()
    graph = build_research_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": run_id}}

    # Check if there is an existing checkpoint
    existing = checkpointer.get_tuple(config)
    state_input = (
        None
        if existing
        else {
            "run_id": run_id,
            "question": topic,
            "start_time": time.time(),
            "total_tokens": 0,
            "total_usd_cost": 0.0,
            "step_index": 0,
        }
    )

    try:
        final_state = graph.invoke(state_input, config=config)
        if final_state and final_state.get("final_report"):
            redis_store.save_final_report(run_id, final_state["final_report"].model_dump_json())
            # Track all executed agent steps to verify no duplicates
            client = redis_store.get_client()
            for trace in final_state.get("trace_log", []):
                agent = trace.agent if hasattr(trace, "agent") else trace.get("agent")
                sub_q = ""
                input_data = trace.input_data if hasattr(trace, "input_data") else trace.get("input_data", {})
                if isinstance(input_data, dict):
                    sub_q = input_data.get("sub_question_id", "")
                call_key = f"{agent}:{sub_q}" if sub_q else agent
                client.rpush(f"chaos:calls:{run_id}", call_key)
            print(f"SUCCESS:{run_id}", flush=True)
    except Exception as e:
        print(f"ERROR:{e}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(1)
    run_id_arg = sys.argv[1]
    topic_arg = sys.argv[2]
    delay = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    run_worker_process(run_id_arg, topic_arg, delay)
