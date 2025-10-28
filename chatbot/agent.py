from dataclasses import dataclass
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import START, StateGraph, END

from prompts import SYSTEM_PROMPT, USER_PROMPT, SUMMARIZE_PROMPT
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from utils import init_chat_llm, init_vector_store
from tools import execute_python_code
import argparse
import yaml

CONFIG = yaml.safe_load(open("./chatbot/config.yaml"))
@dataclass
class State:
    retriever: object = None 
    llm: BaseChatModel = None
    query: str = None
    chat_history: list[dict[str, str]] = None
    context: list[str] = None
    max_tokens: int = 100_000
    current_tokens: int = 0
    last_counted_message: int = 0

def create_initial_state(retriever: object, model="gpt-5-nano", temperature=0.0) -> State:
    llm = init_chat_llm(model, temperature)
    chat_history = [SystemMessage(content=SYSTEM_PROMPT)]
    initial_state = State(
        retriever=retriever, llm=llm, chat_history=chat_history
    )

    return initial_state


##### DEFINITION OF NODES #####

def retrieve(state: State):
    query = state.query
    docs = state.retriever.invoke(query)
    state.context = [doc.page_content for doc in docs]  
    return state


def summarize_chat_hist(state: State):
    user_message = state.chat_history.pop(-1)
    summary = state.llm.invoke(state.chat_history + [HumanMessage(content=SUMMARIZE_PROMPT)])
    state.chat_history = [
        SystemMessage(content=SYSTEM_PROMPT),
        summary,
        HumanMessage(content=user_message),
    ]
    for message in state.chat_history[::-1]:
        
        tokens = state.llm.get_num_tokens(message.content)  # Accurate token counting
        state.current_tokens += tokens
        state.last_counted_message += 1
    return state


def agent_invoke(state: State) -> State:
    context = "\n".join(state.context)
    prompt = USER_PROMPT.format(query=state.query, context=context)
    state.chat_history[-1] = HumanMessage(content=prompt)
    response = state.llm.invoke(state.chat_history)
    state.chat_history.append(response)
    return state


def execute_tool(state: State) -> State:
    last_message = state.chat_history[-1]
    tool_call = last_message.tool_calls[0]

    if tool_call["name"] == "execute_python_code":
        tool_output = execute_python_code.run(tool_call["args"])
        observation_message = ToolMessage(content=f"Tool Output:\n{tool_output}", tool_call_id=tool_call["id"])
        state.chat_history.append(observation_message)
    else:
        raise ValueError(f"Unknown tool: {tool_call['name']}")

    return state


#### DEFINETION OF EDGES / CONDITIONS #####


def is_chat_too_long(state: State) -> bool:
    # TODO: Optimized so that total_tokens is hold in memory and only latests messages are calculated
    last_counted_message = state.last_counted_message if state.last_counted_message > 0 else None
    for message in state.chat_history[:last_counted_message:-1]:
        tokens = state.llm.get_num_tokens(message.content)  # Accurate token counting
        if state.current_tokens + tokens >= state.max_tokens:
            return "summarize_chat_hist"
        state.current_tokens += tokens
        state.last_counted_message += 1 
    return "agent_invoke"


def is_tool_call(state: State) -> bool:
    last_message = state.chat_history[-1]
    if isinstance(last_message, AIMessage) and len(last_message.tool_calls) > 0:
        return "execute_tool"
    return END


#### DEFINITION OF GRAPH AND RUNNER #####


def build_graph() -> StateGraph:
    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("agent_invoke", agent_invoke)
    graph_builder.add_node("summarize_chat_hist", summarize_chat_hist)
    graph_builder.add_node("execute_tool", execute_tool)

    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_conditional_edges("retrieve", is_chat_too_long)
    graph_builder.add_conditional_edges("agent_invoke", is_tool_call)
    graph_builder.add_edge("execute_tool", END)
    graph_builder.add_edge("summarize_chat_hist", "agent_invoke")

    graph = graph_builder.compile()
    return graph


def run_agent(graph: StateGraph, initial_state: State):
    state = initial_state
    while True:
        user_message = input('User: ')
        state.chat_history.append(HumanMessage(content=user_message))
        state.query = user_message
        response = graph.invoke(state)
        print("\nAssistant:", response['chat_history'][-1].content, "\n")
        state.chat_history.append(AIMessage(content=response['chat_history'][-1].content))

if __name__ == "__main__":
    vector_store = init_vector_store()
    initial_state = create_initial_state(vector_store.as_retriever(search_kwargs={"k": CONFIG['n_retrieved_docs']}), model=CONFIG['llm_model'], temperature=CONFIG['temperature'])
    graph = build_graph()
    run_agent(graph, initial_state)
