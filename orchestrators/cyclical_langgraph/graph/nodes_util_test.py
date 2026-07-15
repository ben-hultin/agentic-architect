from graph.nodes_util import analyst_node

def test_analyst_uses_max_cycles_to_continue():
    result = analyst_node({
        "messages": [],
        "active_context": {},
        "routing_flag": "analyze",
        "cycle_count": 3,
        "max_cycles": 5,
        "thread_id": "test",
    })
    assert result["routing_flag"] == "classify"
    assert result["cycle_count"] == 4

def test_analyst_uses_max_cycles_to_end():
    result = analyst_node({
        "messages": [],
        "active_context": {},
        "routing_flag": "analyze",
        "cycle_count": 5,
        "max_cycles": 5,
        "thread_id": "test",
    })
    assert result["routing_flag"] == "end"
    assert result["active_context"]["analysis_complete"] is True

def test_analyst_ignores_hardcoded_three_when_max_cycles_higher():
    result = analyst_node({
        "messages": [],
        "active_context": {},
        "routing_flag": "analyze",
        "cycle_count": 3,
        "max_cycles": 5,
        "thread_id": "test",
    })
    assert result["routing_flag"] != "end"
