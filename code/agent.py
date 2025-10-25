
from dataclasses import dataclass
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import START, StateGraph

from prompts import SYSTEM_PROMPT, USER_PROMPT, SUMMARIZE_PROMPT
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from utils import get_input, init_llm, init_vector_store


@dataclass
class State:
    vector_store: object = None
    llm: BaseChatModel = None
    query: str = None
    chat_history: list[dict[str, str]] = None
    n_retrieved_docs: int = 5
    context: list[str] = None
    max_tokens: int = 100_000


def get_user_input(state: State) -> str:
    # If there is no chat history, provide a default prompt
    assistant_message = "How can I assist you today?\n"

    # If exist chat history, show last assistant message
    if state.chat_history:
        if isinstance(state.chat_history[-1], HumanMessage):
            # If last message is from user, raise error as two inputs have been collected
            raise ValueError(f"Last message in chat history is already from user.{state.chat_history[-1].content}")
        elif isinstance(state.chat_history[-1], AIMessage):
            assistant_message = state.chat_history[-1].content

    user_input = get_input(assistant_message)
    state.query = user_input
    state.chat_history.append(HumanMessage(content=user_input))
    return state


def retriever(state: State):
    query = state.query
    docs = state.vector_store.similarity_search(query, k=state.n_retrieved_docs)
    state.context = [doc.page_content for doc in docs]
    return state


def is_chat_too_long(state: State) -> bool:
    # TODO: Optimized so that total_tokens is hold in memory and only latests messages are calculated
    total_tokens = 0
    for message in state.chat_history:
        tokens = state.llm.get_num_tokens(message.content)  # Accurate token counting
        if total_tokens + tokens >= state.max_tokens:
            return "summarize_chat_hist"
        total_tokens += tokens
    return "agent_invoke"


def summarize_chat_hist(state: State):
    user_message = state.chat_history.pop(-1)
    summary = state.llm.invoke(state.chat_history + [HumanMessage(content=SUMMARIZE_PROMPT)])
    state.chat_history = [
        SystemMessage(content=SYSTEM_PROMPT),
        summary,
        HumanMessage(content=user_message),
    ]
    return state


def agent_invoke(state: State) -> State:
    context = "\n".join(state.context)
    prompt = USER_PROMPT.format(query=state.query, context=context)
    state.chat_history.append(HumanMessage(content=prompt))
    response = state.llm.invoke(state.chat_history)
    state.chat_history.append(response)
    return state


def build_graph() -> StateGraph:
    graph_builder = StateGraph(State)
    graph_builder.add_node("user_input", get_user_input)
    graph_builder.add_node("retriever", retriever)
    graph_builder.add_node("agent_invoke", agent_invoke)
    graph_builder.add_node("summarize_chat_hist", summarize_chat_hist)

    graph_builder.add_edge(START, "user_input")
    graph_builder.add_edge("user_input", "retriever")
    graph_builder.add_conditional_edges("retriever", is_chat_too_long)
    graph_builder.add_edge("summarize_chat_hist", "agent_invoke")
    graph_builder.add_edge("agent_invoke", "user_input")
    # TODO: Escape node?
    graph = graph_builder.compile()
    return graph


def create_initial_state(vector_store: object, n_retrieved_docs: int = 3, model="gpt-5-nano", temperature=0.0) -> State:
    llm = init_llm(model, temperature)
    chat_history = [SystemMessage(content=SYSTEM_PROMPT)]
    initial_state = State(
        vector_store=vector_store, n_retrieved_docs=n_retrieved_docs, llm=llm, chat_history=chat_history
    )

    return initial_state


def run_agent(graph: StateGraph, initial_state: State):
    state = initial_state
    while True:
        state = graph.invoke(state)


if __name__ == "__main__":
    # Example usage
    vector_store = init_vector_store()
    initial_state = create_initial_state(vector_store)
    graph = build_graph()
    run_agent(graph, initial_state)
