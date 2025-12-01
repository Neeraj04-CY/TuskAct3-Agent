from eikon_engine.strategies import plan as planner_v2


def _topological_order(nodes, edges):
    indegree = {node["id"]: 0 for node in nodes}
    adjacency = {node["id"]: [] for node in nodes}
    for src, dst in edges:
        indegree[dst] += 1
        adjacency[src].append(dst)

    level = [node_id for node_id, deg in indegree.items() if deg == 0]
    order = []
    while level:
        current = level.pop(0)
        order.append(current)
        for neighbor in adjacency[current]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                level.append(neighbor)
    return order


def test_planner_v2_generates_valid_dag():
    goal = "Analyze quarterly data and summarize findings"
    plan = planner_v2(goal)

    nodes = plan["nodes"]
    edges = plan["edges"]

    assert 2 <= len(nodes) <= 4
    assert plan["goal"] == goal

    node_ids = [node["id"] for node in nodes]
    assert len(node_ids) == len(set(node_ids))

    order = _topological_order(nodes, edges)
    assert len(order) == len(nodes)

    last_id = nodes[-1]["id"]
    if len(nodes) > 1:
        destinations = {dst for _, dst in edges}
        assert destinations == {last_id}
        sources = {src for src, _ in edges}
        assert sources == set(node_ids) - {last_id}