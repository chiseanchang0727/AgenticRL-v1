from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes import PlannerNode, CandidateRoomFilterNode, RoomSelectorNode, JudgeNode


def build_agent(state, verl_replacement):

    graph = StateGraph(state)
    planner = PlannerNode(verl_replacement=verl_replacement)
    candidate_room_filter = CandidateRoomFilterNode(verl_replacement=verl_replacement)
    room_selector = RoomSelectorNode(verl_replacement=verl_replacement)
    judge = JudgeNode(verl_replacement=verl_replacement)

    # Register nodes
    graph.add_node("planner", planner)
    graph.add_node("candidate_room_filter", candidate_room_filter)
    graph.add_node("room_selector", room_selector)
    graph.add_node("judge", judge)
    # graph.add_node("Answer", answer_node)

    # Entry
    graph.add_edge(START, "candidate_room_filter")
    graph.add_edge("candidate_room_filter", "room_selector")
    
    # TODO: add a LLM selector and train it with GRPO
    graph.add_edge("room_selector", END)
    

    # Compile
    app = graph.compile()

    return app

