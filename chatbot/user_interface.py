import gradio as gr
from agent import create_initial_state, init_vector_store, build_graph
from langchain.messages import HumanMessage


def build_interface():
    vector_store = init_vector_store()
    initial_state = create_initial_state(vector_store)
    graph = build_graph()

    def run(message, history):
        state = initial_state
        state.query = message

        # Append the new user message
        state.chat_history.append(HumanMessage(content=message))

        response = graph.invoke(state)

        return response["chat_history"][-1].content

    return run


if __name__ == "__main__":
    run = build_interface()
    gr.ChatInterface(fn=run, type="messages").launch()
