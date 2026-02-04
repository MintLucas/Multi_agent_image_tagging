from langgraph.types import interrupt
from typing import TypedDict
from langgraph.types import Command
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
# 定义图状态
class State(TypedDict):
    action_details: str  # 待审批的操作详情

def approval_node(state: State):
    # 触发暂停，传递审批所需信息（负载）
    is_approved = interrupt({
        "question": "是否批准该操作？",
        "details": state["action_details"]  # 让人工看到具体操作内容
    })
    # 恢复后，is_approved 会接收 Command(resume=...) 传入的值
    return {"approved": is_approved}

# 构建 LangGraph 图
graph = StateGraph(State)
graph.add_node("approval_node", approval_node)


graph.add_edge(START, "approval_node")
graph.add_edge("approval_node", END)  # 无条件边：执行完成 → 结束

graph = graph.compile(checkpointer=checkpointer)

# 1. 初始运行（触发暂停）
config = {"configurable": {"thread_id": "flow-123"}}  # 流程唯一标识
initial_result = graph.invoke({"action_details": "转账500元"}, config=config)

# 查看暂停信息（__interrupt__ 字段包含中断负载）
print(initial_result)
print(initial_result["__interrupt__"])  # 输出：[Interrupt(value={'question': ..., 'details': ...})]

# # 2. 恢复执行（传入人工审批结果）
resumed_result = graph.invoke(Command(resume="yes"), config=config)  # resume=True 表示批准
# print(resumed_result)  # 输出：True
print(resumed_result["approved"])  # 输出：True