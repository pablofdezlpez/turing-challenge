from typing import TypedDict
from openai import OpenAI
from langgraph.graph import START, StateGraph
import dotenv
import os
dotenv.load_dotenv()

class State(TypedDict):
    query: str
    chat_history: list[dict[str, str]]
    vector_store: object
    n_retrieved_docs: int
    context: list[str]

def init_llm(api_key: str = None, model: str = "gpt-4o-550k-2", temperature: float = 0.0) -> OpenAI:
    if api_key is None:
        raise ValueError("API key must be provided")
    llm = OpenAI(api_key=api_key, model=model, temperature=temperature)
    return llm

def get_input(text: str) -> str:
    return input(text)
    
def get_user_input(state: State) -> str:
    llm_response = state['chat_history'][-1]['assistant'] if state['chat_history'] else "How can I assist you today?"
    user_input = get_input(llm_response)
    state['query'] = user_input
    state['chat_history'].append({"role": "user", "content": user_input})
    return state

def retriever(state: State):
    query = state['query']
    docs = state['vector_store'].similarity_search(query, k=state['n_retrieved_docs'])
    state['context'] = [doc.page_content for doc in docs]
    return state

def agent_invoke(state: State, llm: OpenAI) -> State:
    context = "\n".join(state['context'])
    prompt = f"Context:\n{context}\n\nUser Query:\n{state['query']}\n\nAnswer:"
    system_prompt = "You are a helpful assistant."
    response = llm.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    answer = response.choices[0].message['content']
    state.append({'role': 'assistant', 'content': answer})
    return state

def build_graph() -> StateGraph:
    graph_builder = StateGraph(State)
    graph_builder.add_node('user_input', get_user_input)
    graph_builder.add_node('retriever', retriever)
    graph_builder.add_node('agent_invoke', agent_invoke)

    graph_builder.add_edge(START, "user_input")
    graph_builder.add_edge("user_input", "retriever")
    graph_builder.add_edge("retriever", "agent_invoke")
    graph_builder.add_edge("agent_invoke", "user_input")
    #TODO: Escape node?
    graph = graph_builder.compile()
    return graph

def create_initial_state(vector_store: object, n_retrieved_docs: int = 3) -> State:
    initial_state: State = {
        "query": "",
        "chat_history": [],
        "vector_store": vector_store,
        "n_retrieved_docs": n_retrieved_docs,
        "context": []
    }
    return initial_state

def run_agent(graph: StateGraph, initial_state: State, llm: OpenAI):
    state = initial_state
    while True:
        state = graph.run(state, llm=llm)

if __name__ == "__main__":
    # Example usage
    agent = init_llm(api_key=os.getenv("OPENAI_API_KEY"))
    vector_store = ...  # Initialize your vector store here
    initial_state = create_initial_state(vector_store)
    graph = build_graph()
    run_agent(graph, initial_state, agent)
