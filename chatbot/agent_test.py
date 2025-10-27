import pytest
from langchain.messages import AIMessage
from agent import is_chat_too_long, State, init_vector_store, create_initial_state, retrieve, is_tool_call
from langgraph.graph import END


def test_retrieve_returns_n_documents():
    vector_store = init_vector_store()
    state = create_initial_state(vector_store)
    state.num_retrieved_documents = 3
    state.query = "sample query"
    state = retrieve(state)
    assert len(state.context) == 3


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


def test_is_tool_call(mocker):
    mock_tool_response = mocker.Mock()
    mock_tool_response.content = "Tool executed successfully"
    vector_store = init_vector_store()
    state = create_initial_state(vector_store)
    state.chat_history.append(
        AIMessage(
            content="Executing tool",
            tool_calls=[
                {"name": "execute_python_code", "args": {"code": "print('Hello World')", "id": "1"}, "id": "1"}
            ],
        )
    )

    routing = is_tool_call(state)

    assert routing == "execute_tool"


def test_is_not_tool_call(mocker):
    mock_tool_response = mocker.Mock()
    mock_tool_response.content = "Tool executed successfully"
    vector_store = init_vector_store()
    state = create_initial_state(vector_store)
    state.chat_history.append(AIMessage(content="Not Executing tool"))

    routing = is_tool_call(state)

    assert routing == END
