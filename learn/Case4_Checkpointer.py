from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]

def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}


workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}
graph.invoke({"foo": ""}, config)

config = {"configurable": {"thread_id": "1"}}
print(graph.get_state(config))

# # get a state snapshot for a specific checkpoint_id
# config = {"configurable": {"thread_id": "1", "checkpoint_id": "1f0cdb4c-d83e-6199-8002-f5d5b572280f"}}
# print(graph.get_state(config))


# config = {"configurable": {"thread_id": "1"}}
# print(list(graph.get_state_history(config)))

# config = {"configurable": {"thread_id": "1", "checkpoint_id": "0c62ca34-ac19-445d-bbb0-5b4984975b2a"}}
# graph.invoke(None, config=config)

graph.update_state(config, {"foo": 2, "bar": ["b"]})
print(graph.get_state(config))