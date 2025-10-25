import pytest
from agent import is_chat_too_long, create_initial_state, build_graph, run_agent, State, get_user_input


def test_get_user_input_initial_conversation(mocker):
    mocker.patch("agent.get_input", return_value="an user input")
    state = create_initial_state(vector_store=None)

    state = get_user_input(state)
    history = state.chat_history
    
    assert state.query == "an user input"
    assert history[-1]["role"] == "user"
    assert history[-1]["content"] == "an user input"

    return state


def test_gest_user_input_subsequent_conversation(mocker):
    mocker.patch("agent.get_input", return_value="another user input")
    state = State(chat_history=[{"role": "assistant", "content": "latest response"}])

    state = get_user_input(state)
    history = state.chat_history

    assert state.query == "another user input"
    assert history[1]["role"] == "user"
    assert history[1]["content"] == "another user input"
    assert history[0]["role"] == "assistant"
    assert history[0]["content"] == "latest response"

    return state


@pytest.mark.parametrize("max_tokens, current_tokens, expected", [(5, 1, False), (5, 5, True), (5, 10, True)])
def test_is_chat_too_long(mocker, max_tokens, current_tokens, expected):
    mock_llm = mocker.Mock()
    mock_llm.get_num_tokens.return_value = 1
    state = State(
        llm=mock_llm,
        chat_history=[{"role": "assistant", "content": "latest response"}] * current_tokens,
        max_tokens=max_tokens,
    )

    response = is_chat_too_long(state)

    assert response == expected
