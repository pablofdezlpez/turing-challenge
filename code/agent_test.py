import pytest
from agent import init_llm, create_initial_state, build_graph, run_agent, State, get_user_input
    
def test_get_user_input_initial_conversation(mocker):
    mocker.patch("agent.get_input", return_value="an user input")
    state = create_initial_state(vector_store=None)
    state = get_user_input(state)
    history = state['chat_history']
    assert state['query'] == "an user input"
    assert len(history) == 1
    assert history[0]['role'] == 'user'
    assert history[0]['content'] == "an user input"

    return state
