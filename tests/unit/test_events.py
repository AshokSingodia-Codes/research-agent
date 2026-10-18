import pytest

from app.api.events import EventBroker, register_stream_queue, unregister_stream_queue


@pytest.mark.unit
def test_event_broker_registration_and_cleanup():
    """Validates queue registration and unregistration."""
    broker = EventBroker()
    run_id = "test_run_events_reg"

    q1 = broker.register_queue(run_id)
    assert run_id in broker._subscribers
    assert q1 in broker._subscribers[run_id]

    broker.register_queue(run_id)
    assert len(broker._subscribers[run_id]) == 2

    # Unregister specific queue
    broker.unregister_queue(run_id, q1)
    assert len(broker._subscribers[run_id]) == 1
    assert q1 not in broker._subscribers[run_id]

    # Unregister all
    broker.unregister_queue(run_id)
    assert run_id not in broker._subscribers


@pytest.mark.unit
def test_event_broker_publish_event():
    """Validates publishing an event increments sequence, stores in history, and pushes to subscribers."""
    broker = EventBroker()
    run_id = "test_run_publish"
    q = broker.register_queue(run_id)

    broker.publish_event(run_id, "step_started", {"step": "planning"})

    # Check queue got the event
    assert not q.empty()
    item = q.get_nowait()
    assert item["event"] == "step_started"
    assert item["data"] == {"step": "planning"}
    assert item["id"] >= 1

    broker.unregister_queue(run_id, q)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_event_generator_replay_and_stream():
    """Validates historical event replay with Last-Event-ID and live completion."""
    broker = EventBroker()
    run_id = "test_run_replay_stream"

    # Pre-populate history
    broker.publish_event(run_id, "node_start", {"node": "planner"})
    broker.publish_event(run_id, "node_complete", {"node": "planner"})
    broker.publish_event(run_id, "completed", {"report": "final"})

    # Generator without last_event_id: gets all 3 events
    events = []
    async for chunk in broker.event_generator(run_id, last_event_id=None):
        events.append(chunk)

    assert len(events) == 3
    assert "event: node_start" in events[0]
    assert "event: node_complete" in events[1]
    assert "event: completed" in events[2]

    # Replay with last_event_id=1: should only get events 2 and 3
    replayed = []
    async for chunk in broker.event_generator(run_id, last_event_id=1):
        replayed.append(chunk)

    assert len(replayed) == 2
    assert "event: node_complete" in replayed[0]
    assert "event: completed" in replayed[1]


@pytest.mark.unit
def test_stream_queue_helpers():
    """Validates helper functions ported from src."""
    run_id = "test_run_helper"
    q = register_stream_queue(run_id)
    assert q is not None
    unregister_stream_queue(run_id, q)
