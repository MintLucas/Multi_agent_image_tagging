from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

class State(TypedDict):
    user_input: str
    step: int = 1

def step1(state: State):
    print(f"执行步骤1，当前输入：{state['user_input']}")
    # 满足条件时主动中断（比如需要用户补充信息）
    if "more" in state['user_input'].lower():
        interrupt(value="more in user_input")  # 主动暂停，保存检查点
    return {"step": 2}

def step2(state: State):
    print(f"执行步骤2，当前步骤：{state['step']}")
    return {"step": 3}

# 构建工作流+绑定checkpointer
builder = StateGraph(State)
builder.add_node("step1", step1)
builder.add_node("step2", step2)
builder.add_edge(START, "step1")
builder.add_edge("step1", "step2")
builder.add_edge("step2", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 1. 第一次执行：主动暂停
thread_id = "test_thread"
config = {"configurable": {"thread_id": thread_id}}
# 输入包含"more"，触发step1中的中断
print(graph.invoke({"user_input": "I need more info"}, config))  # 执行step1后暂停

# 2. 恢复执行：用Command原语，传入相同thread_id和更新后的状态
updated_state = {"user_input": "I need info about pizza"}  # 更新状态
print(graph.invoke(command=Command(resume=True), config=config, input=updated_state))
# 输出：执行步骤2，当前步骤：2（从step1暂停处继续）