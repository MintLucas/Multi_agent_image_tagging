from typing import List, Optional
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, END, START

# 1. 定义共享状态（无修改）
class HITLState(BaseModel):
    messages: List[BaseMessage] = []
    draft_plan: Optional[str] = None
    human_decision: Optional[str] = None
    revised_plan: Optional[str] = None

# 2. Agent 节点：生成初步方案（无修改）
def plan_agent(state: HITLState) -> HITLState:
    print("\n=== Agent 正在生成方案 ===")
    draft = "初步方案：在小红书+抖音投放产品广告，预算5000元，为期1周"
    state.draft_plan = draft
    state.messages.append(AIMessage(content=f"Agent 生成初稿：{draft}"))
    print(f"Agent 输出：{draft}\n")
    return state

# 3. 人工介入节点（无修改）
def human_review(state: HITLState) -> HITLState:
    print("=== 【人工介入】请审核 Agent 方案 ===")
    print(f"当前初稿：{state.draft_plan}")
    print("\n请输入你的决策（输入对应关键词后回车）：")
    print("  1. approve → 批准方案，继续执行")
    print("  2. revise  → 修改方案（需输入修改后的内容）")
    print("  3. abort   → 终止流程")

    while True:
        decision = input("\n你的决策（approve/revise/abort）：").strip().lower()
        if decision not in ["approve", "revise", "abort"]:
            print("输入无效！请重新输入上述关键词")
            continue

        state.human_decision = decision
        if decision == "revise":
            revised = input("请输入修改后的方案：").strip()
            state.revised_plan = revised
            state.messages.append(HumanMessage(content=f"人工修改方案：{revised}"))
            print(f"\n已接收你的修改：{revised}")
        elif decision == "approve":
            state.messages.append(HumanMessage(content="人工批准方案，继续执行"))
            print("\n你已批准方案！")
        else:  # abort
            state.messages.append(HumanMessage(content="人工终止流程"))
            print("\n你已终止流程！")
        break

    return state

# 4. 执行节点（无修改）
def execute_agent(state: HITLState) -> HITLState:
    print("\n=== Agent 正在执行最终方案 ===")
    if state.human_decision == "approve":
        final_plan = state.draft_plan
    else:  # revise
        final_plan = state.revised_plan

    execution_result = f"执行成功！最终方案：{final_plan}\n执行动作：1. 联系小红书达人；2. 抖音投放开户；3. 预算锁定"
    state.messages.append(AIMessage(content=execution_result))
    print(execution_result)
    return state

# 5. 路由函数（无修改，但使用方式变了）
def decision_router(state: HITLState) -> str:
    if state.human_decision == "abort":
        return END  # 终止流程
    else:
        return "execute_agent"  # 路由到执行节点

# 6. 构建 LangGraph 图（关键修改：用 add_conditional_edges 绑定路由函数）
graph = StateGraph(HITLState)

# 添加节点（无修改）
graph.add_node("plan_agent", plan_agent)
graph.add_node("human_review", human_review)
graph.add_node("execute_agent", execute_agent)


graph.add_edge(START, "plan_agent")
graph.add_edge("plan_agent", "human_review") 
graph.add_conditional_edges(
    "human_review",  # 源节点：人工审核
    decision_router,   # 路由函数（返回目标节点名称）
    ["execute_agent", END]
)
graph.add_edge("execute_agent", END)  # 无条件边：执行完成 → 结束


app = graph.compile()

png_data = app.get_graph().draw_mermaid_png()
with open("hitl_flowchart.png", "wb") as f:
    f.write(png_data)

# 7. 运行流程（无修改）
if __name__ == "__main__":
    print("=== 启动 Human-in-the-Loop 流程 ===")
    final_state = app.invoke({
        "messages": [HumanMessage(content="请制定一个产品推广方案")]
    })
    print("\n=== 流程结束 ===")
    print("\n📝 交互消息历史：")
    for msg in final_state["messages"]:
        msg.pretty_print()