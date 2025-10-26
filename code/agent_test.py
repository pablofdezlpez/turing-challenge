import pytest
from langchain.messages import AIMessage
from agent import is_chat_too_long, State

@pytest.mark.parametrize(
    "max_tokens, current_tokens, expected",
    [(5, 1, "agent_invoke"), (5, 5, "summarize_chat_hist"), (5, 10, "summarize_chat_hist")],
)
def test_is_chat_too_long(mocker, max_tokens, current_tokens, expected):
    mock_llm = mocker.Mock()
    mock_llm.get_num_tokens.return_value = 1
    state = State(
        llm=mock_llm,
        chat_history=[AIMessage(content="latest response")] * current_tokens,
        max_tokens=max_tokens,
    )

    response = is_chat_too_long(state)

    assert response == expected
