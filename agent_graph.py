import json
import operator
import re
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from tools import get_part_id, get_shipping_cost, get_stock_level, get_supplier_location

MODEL_NAME = "granite4:tiny-h" 
llm = ChatOllama(model=MODEL_NAME, temperature=0, num_ctx=10240)

def inventory_node(state):
    messages = state['messages']
    sys_msg = SystemMessage(content="""You are the Inventory Manager. Your sole purpose is to handle requests related to **Part IDs** and **Stock Levels**.
    
    You have access to two tools: `get_part_id` and `get_stock_level`.
    
    RULES:
    1. Always use `get_part_id` first if the user provides a common part name (e.g., Engine). You cannot check stock without the ID.
    2. After gathering all necessary information using the tools, state the final answer clearly to the Supervisor/User.
    3. Do NOT provide multi-step calculations or reasoning. Just use the tools and output the result.

    FEW-SHOT EXAMPLE:
    User: How many Windshields are in stock?
    Worker: 
    
    Call: get_part_id("Windshield")
    Result: ID-555
    
    Call: get_stock_level("ID-555")
    Result: 15
    
    Final Answer: There are 15 Windshields in stock.

    FEW-SHOT EXAMPLE (MULTI-STEP):
    User: How many Engines do we have?
    Worker: 
    
    # Step 1: Find ID
    Call: get_part_id("Engine")
    Result: ID-999
    
    # Step 2: Check Stock
    Call: get_stock_level("ID-999")
    Result: 4
    
    Final Answer: We have 4 Engines in stock.
    """)
    
    tools = [get_part_id, get_stock_level]
    llm_with_tools = llm.bind_tools(tools) 
    
    response = llm_with_tools.invoke([sys_msg] + messages) 
    return {"messages": [response]}

def logistics_node(state):
    messages = state['messages']
    sys_msg = SystemMessage(content="""You are the Logistics Manager. Your sole purpose is to handle requests related to **Supplier Locations** and **Shipping Costs**.
    
    You have access to two tools: `get_supplier_location` and `get_shipping_cost`.
    
    RULES:
    1. If the user asks for shipping cost for a part, you must first find the supplier location using `get_supplier_location`.
    2. After gathering all necessary information using the tools, state the final answer clearly to the Supervisor/User.
    3. Do NOT provide multi-step calculations or reasoning. Just use the tools and output the result.

    FEW-SHOT EXAMPLE:
    User: What is the cost to ship ID-200?
    Worker: 
    
    Call: get_supplier_location("ID-200")
    Result: Berlin
    
    Call: get_shipping_cost("Berlin")
    Result: 60 EUR
    
    Final Answer: The shipping cost for ID-200 is 60 EUR.

    FEW-SHOT EXAMPLE (MULTI-STEP):
    User: How much to ship for a Tire? (The Supervisor already found ID-100)
    Worker: 
    
    Call: get_supplier_location("ID-100")
    Result: Munich
    
    Call: get_shipping_cost("Munich")
    Result: 50 EUR
    
    Final Answer: The shipping cost for the Tire supplier is 50 EUR.
    """)
    
    tools = [get_supplier_location, get_shipping_cost]
    llm_with_tools = llm.bind_tools(tools)
    
    response = llm_with_tools.invoke([sys_msg] + messages)
    return {"messages": [response]} 

all_tools = [get_part_id, get_stock_level, get_supplier_location, get_shipping_cost]
tool_node = ToolNode(all_tools)

class SupervisorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

def supervisor_node(state: SupervisorState):
    messages = state['messages']
    last_user_message = messages[0].content
    
    system_prompt = """You are the SCM Supervisor. You manage two workers:
    1. Inventory_Worker: Handles Part IDs and Stock Checks.
    2. Logistics_Worker: Handles Supplier Cities and Shipping Costs.

    YOUR JOB:
    - Analyze the user's request.
    - Decide which worker should act next.
    - If the user's request is a greeting ("Hello"), or an out-of-scope question (e.g., weather, politics, non-SCM topics), choose 'FINISH'.

    OUTPUT FORMAT: JSON ONLY.
    {"next": "Inventory_Worker"} 
    OR 
    {"next": "Logistics_Worker"} 
    OR 
    {"next": "FINISH"}
    """
    
    supervisor_messages = [{"role": "system", "content": system_prompt}] + messages
    response = llm.invoke(supervisor_messages)
    
    try:
        content = response.content
        match = re.search(r"(\{.*\})", content, re.DOTALL)
        if match:
            decision = json.loads(match.group(1))
            return {"next": decision.get("next", "FINISH")}
    except:
        pass
        
    lower_content = response.content.lower()
    if "inventory" in lower_content: return {"next": "Inventory_Worker"}
    if "logistics" in lower_content: return {"next": "Logistics_Worker"}
    return {"next": "FINISH"}

# Graph
workflow = StateGraph(SupervisorState)

workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Inventory_Worker", inventory_node)
workflow.add_node("Logistics_Worker", logistics_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("Supervisor")

workflow.add_conditional_edges(
    "Supervisor",
    lambda x: x["next"], 
    {
        "Inventory_Worker": "Inventory_Worker",
        "Logistics_Worker": "Logistics_Worker",
        "FINISH": END
    }
)

def should_continue(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "Supervisor"

workflow.add_conditional_edges("Inventory_Worker", should_continue, {"tools": "tools", "Supervisor": "Supervisor"})
workflow.add_conditional_edges("Logistics_Worker", should_continue, {"tools": "tools", "Supervisor": "Supervisor"})

workflow.add_edge("tools", "Supervisor")

app = workflow.compile()

def run_hierarchical_agent(query: str):
    initial_state = {"messages": [HumanMessage(content=query)]}
    result = app.invoke(initial_state, {"recursion_limit": 20})
    return result["messages"][-1].content