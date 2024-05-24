import pytest
from unittest.mock import patch, MagicMock

from totokenizers.google import GoogleTokenizer
from totokenizers.factories import Totokenizer
from totokenizers.schemas import (
    ChatMLMessage,
    FunctionCall,
    FunctionCallChatMLMessage,
    FunctionChatMLMessage,
    Chat,
)
from pytest_mock import MockerFixture


@pytest.fixture(scope="module")
def model_name():
    return "gemini-1.0-pro-002"


@pytest.fixture(scope="module")
def model_tag(model_name: str):
    return f"google/{model_name}"


@pytest.fixture(scope="module")
def chatml_messages():
    return [
        ChatMLMessage(
            content="You are a bot.",
            name="system",
            role="system",
        ),
        ChatMLMessage(
            content="hello bot",
            name="user",
            role="user",
        ),
        ChatMLMessage(
            content="I am Skynet.",
            name="skynet",
            role="assistant",
        ),
    ]


@pytest.fixture(scope="module")
def function_call_chat_ml_message():
    return FunctionCallChatMLMessage(
        **{
            "function_call": FunctionCall(
                **{
                    "name": "get_current_weather",
                    "arguments": '{"location": "Boston, MA"}',
                }
            ),
            "role": "assistant",
        }
    )


@pytest.fixture(scope="module")
def function_chat_ml_message():
    return FunctionChatMLMessage(
        content='{"temperature": "22", "unit": "celsius", "description": "Sunny"}',
        name="get_current_weather",
        role="function",
    )


@pytest.fixture(scope="module")
def tools():
    return [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        }
    ]


@pytest.fixture(scope="module")
def chat():
    return [
        ChatMLMessage(content="What is the weather like in Boston?", role="user"),
        FunctionCallChatMLMessage(
            **{
                "function_call": FunctionCall(
                    **{
                        "name": "get_current_weather",
                        "arguments": '{"location": "Boston, MA"}',
                    }
                ),
                "role": "assistant",
            }
        ),
        FunctionChatMLMessage(
            content='{"temperature": "22", "unit": "celsius", "description": "Sunny"}',
            name="get_current_weather",
            role="function",
        ),
    ]


@pytest.fixture(scope="module")
def messages(chat):
    return {
        "messages": chat,
        "functions": [
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            }
        ],
    }


def authentication_mock(*args, **kwargs): ...


def test_count_tokens_gem(mocker, model_tag: str):
    # Create a mock response
    mock_response = MagicMock()
    mock_response = 2
    mocker.patch("totokenizers.google.GoogleTokenizer.load_gemini", authentication_mock)
    # Patch the count_tokens method in the GoogleTokenizer class
    with patch.object(Totokenizer, "count_tokens", return_value=mock_response):
        tokenizer = Totokenizer.from_model(model_tag)
        message = "hello world"
        assert tokenizer.count_tokens(message) == 2


def test_gemini_chatml_messages(mocker, chatml_messages):
    # Create mock responses
    mock_count_chatml_tokens_response = 12
    mock_count_tokens_response = MagicMock()
    mock_count_tokens_response = 2
    mocker.patch("totokenizers.google.GoogleTokenizer.load_gemini", authentication_mock)

    # Patch the count_chatml_tokens and count_tokens methods in the GoogleTokenizer class
    with patch.object(
        GoogleTokenizer,
        "count_chatml_tokens",
        return_value=mock_count_chatml_tokens_response,
    ), patch.object(
        GoogleTokenizer, "count_tokens", return_value=mock_count_tokens_response
    ):

        tokenizer = GoogleTokenizer("gemini-1.0-pro-002")

        # Test count_chatml_tokens
        count_tokens = tokenizer.count_chatml_tokens(chatml_messages)
        assert count_tokens == 12

        # Test count_tokens
        count_tokens = tokenizer.count_tokens("hello world")
        assert count_tokens == 2


def test_gemini_function_call_chat_ml_message(mocker, function_call_chat_ml_message):
    # Create a mock response for count_message_tokens
    mock_count_message_tokens_response = 9
    mocker.patch("totokenizers.google.GoogleTokenizer.load_gemini", authentication_mock)

    # Patch the count_message_tokens method in the GoogleTokenizer class
    with patch.object(
        GoogleTokenizer,
        "count_message_tokens",
        return_value=mock_count_message_tokens_response,
    ):
        tokenizer = GoogleTokenizer(model_name="gemini-1.0-pro-002")

        count_tokens = tokenizer.count_message_tokens(function_call_chat_ml_message)
        assert count_tokens == 9


def test_gemini_function_chat_ml_message(mocker, function_chat_ml_message):
    # Create a mock response for count_message_tokens
    mock_count_message_tokens_response = 26
    mocker.patch("totokenizers.google.GoogleTokenizer.load_gemini", authentication_mock)
    # Patch the count_message_tokens method in the GoogleTokenizer class
    with patch.object(
        GoogleTokenizer,
        "count_message_tokens",
        return_value=mock_count_message_tokens_response,
    ):
        tokenizer = GoogleTokenizer(model_name="gemini-1.0-pro-002")

        count_tokens = tokenizer.count_message_tokens(function_chat_ml_message)
        assert count_tokens == 26


def test_gemini_tools(mocker, tools):
    # Create a mock response for count_functions_tokens
    mock_count_functions_tokens_response = 28
    mocker.patch("totokenizers.google.GoogleTokenizer.load_gemini", authentication_mock)
    # Patch the count_functions_tokens method in the GoogleTokenizer class
    with patch.object(
        GoogleTokenizer,
        "count_functions_tokens",
        return_value=mock_count_functions_tokens_response,
    ):
        tokenizer = GoogleTokenizer(model_name="gemini-1.0-pro-002")

        count_tokens = tokenizer.count_functions_tokens(tools)
        assert count_tokens == 28


def test_gemini_chat(mocker, messages):
    # Create a mock response for count_chatml_tokens
    mock_count_chatml_tokens_response = 71
    mocker.patch("totokenizers.google.GoogleTokenizer.load_gemini", authentication_mock)
    # Patch the count_chatml_tokens method in the GoogleTokenizer class
    with patch.object(
        GoogleTokenizer,
        "count_chatml_tokens",
        return_value=mock_count_chatml_tokens_response,
    ):
        tokenizer = GoogleTokenizer(model_name="gemini-1.0-pro-002")

        count_tokens = tokenizer.count_chatml_tokens(**messages)
        assert count_tokens == 71


def test_simple_user_message(mocker):
    # Create a mock response for count_message_tokens
    mock_count_message_tokens_response = 5
    mocker.patch("totokenizers.google.GoogleTokenizer.load_gemini", authentication_mock)
    # Patch the count_message_tokens method in the GoogleTokenizer class
    with patch.object(
        GoogleTokenizer,
        "count_message_tokens",
        return_value=mock_count_message_tokens_response,
    ):
        tokenizer = GoogleTokenizer(model_name="gemini-1.0-pro-002")
        user_message_example: ChatMLMessage = {
            "content": "Call the example function.",
            "role": "user",
        }
        assert tokenizer.count_message_tokens(user_message_example) == 5


def test_simple_chat(mocker):
    # Create a mock response for count_chatml_tokens
    mock_count_chatml_tokens_response = 5
    mocker.patch("totokenizers.google.GoogleTokenizer.load_gemini", authentication_mock)
    # Patch the count_chatml_tokens method in the GoogleTokenizer class
    with patch.object(
        GoogleTokenizer,
        "count_chatml_tokens",
        return_value=mock_count_chatml_tokens_response,
    ):
        tokenizer = GoogleTokenizer(model_name="gemini-1.0-pro-002")
        simple_chat: Chat = [
            {"content": "Good bot.", "role": "system"},
            {"content": "Hello.", "role": "user"},
        ]
        assert tokenizer.count_chatml_tokens(simple_chat) == 5


def test_functions_chat_systemless(mocker, example_function_jsonschema: dict):
    # Create a mock response for count_chatml_tokens
    mock_count_chatml_tokens_response = 25
    mocker.patch("totokenizers.google.GoogleTokenizer.load_gemini", authentication_mock)
    # Patch the count_chatml_tokens method in the GoogleTokenizer class
    with patch.object(
        GoogleTokenizer,
        "count_chatml_tokens",
        return_value=mock_count_chatml_tokens_response,
    ):
        tokenizer = GoogleTokenizer(model_name="gemini-1.0-pro-002")
        simple_chat: Chat = [{"content": "Hello.", "role": "user"}]
        functions = [example_function_jsonschema]
        assert (
            tokenizer.count_chatml_tokens(messages=simple_chat, functions=functions)
            == 25
        )


def test_2_functions_chat(
    example_function_jsonschema: dict, example_function2_jsonschema: dict, mocker
):
    # Create a mock response for count_chatml_tokens
    mock_count_chatml_tokens_response = 68
    mocker.patch("totokenizers.google.GoogleTokenizer.load_gemini", authentication_mock)
    # Patch the count_chatml_tokens method in the GoogleTokenizer class
    with patch.object(
        GoogleTokenizer,
        "count_chatml_tokens",
        return_value=mock_count_chatml_tokens_response,
    ):
        tokenizer = GoogleTokenizer(model_name="gemini-1.0-pro-002")
        simple_chat: Chat = [
            {"content": "Good bot.", "role": "system"},
            {"content": "Hello.", "role": "user"},
        ]
        functions = [example_function_jsonschema, example_function2_jsonschema]
        assert tokenizer.count_chatml_tokens(simple_chat, functions) == 68


def test_2_functions_chat_systemless(
    example_function_jsonschema: dict, example_function2_jsonschema: dict, mocker
):
    # Create a mock response for count_chatml_tokens
    mock_count_chatml_tokens_response = 65
    mocker.patch("totokenizers.google.GoogleTokenizer.load_gemini", authentication_mock)
    # Patch the count_chatml_tokens method in the GoogleTokenizer class
    with patch.object(
        GoogleTokenizer,
        "count_chatml_tokens",
        return_value=mock_count_chatml_tokens_response,
    ):
        tokenizer = GoogleTokenizer(model_name="gemini-1.0-pro-002")
        simple_chat: Chat = [{"content": "Hello.", "role": "user"}]
        functions = [example_function_jsonschema, example_function2_jsonschema]
        assert tokenizer.count_chatml_tokens(simple_chat, functions) == 65


def test_function_call_chat(example_function_jsonschema: dict, mocker):
    # Create a mock response for count_chatml_tokens
    mock_count_chatml_tokens_response = 54
    mocker.patch("totokenizers.google.GoogleTokenizer.load_gemini", authentication_mock)
    # Patch the count_chatml_tokens method in the GoogleTokenizer class
    with patch.object(
        GoogleTokenizer,
        "count_chatml_tokens",
        return_value=mock_count_chatml_tokens_response,
    ):
        tokenizer = GoogleTokenizer(model_name="gemini-1.0-pro-002")
        simple_chat = [
            {"content": "Debug bot.", "role": "system"},
            {"content": "Call the example function.", "role": "user"},
            {
                "content": None,
                "function_call": {
                    "name": "exampleFunction",
                    "arguments": '{\n  "param1": "Hello",\n  "param2": 123\n}',
                },
                "role": "assistant",
            },
            {
                "content": "example return: ('Hello', 42)",
                "name": "exampleFunction",
                "role": "function",
            },
        ]
        functions = [example_function_jsonschema]
        assert tokenizer.count_chatml_tokens(simple_chat, functions) == 54


def test_function_call_chat_systemless(example_function_jsonschema: dict, mocker):
    # Create a mock response for count_chatml_tokens
    mock_count_chatml_tokens_response = 51
    mocker.patch("totokenizers.google.GoogleTokenizer.load_gemini", authentication_mock)
    # Patch the count_chatml_tokens method in the GoogleTokenizer class
    with patch.object(
        GoogleTokenizer,
        "count_chatml_tokens",
        return_value=mock_count_chatml_tokens_response,
    ):
        tokenizer = GoogleTokenizer(model_name="gemini-1.0-pro-002")
        simple_chat = [
            {"content": "Call the example function.", "role": "user"},
            {
                "content": None,
                "function_call": {
                    "name": "exampleFunction",
                    "arguments": '{\n  "param1": "Hello",\n  "param2": 123\n}',
                },
                "role": "assistant",
            },
            {
                "content": "example return: ('Hello', 42)",
                "name": "exampleFunction",
                "role": "function",
            },
        ]
        functions = [example_function_jsonschema]
        assert tokenizer.count_chatml_tokens(simple_chat, functions) == 51


def test_function_role(example_function_jsonschema: dict, mocker):
    # Create mock responses for count_functions_tokens and count_chatml_tokens
    mock_count_functions_tokens_response = 23
    mock_count_chatml_tokens_response = 39
    mocker.patch("totokenizers.google.GoogleTokenizer.load_gemini", authentication_mock)
    # Patch the count_functions_tokens method in the GoogleTokenizer class
    with patch.object(
        GoogleTokenizer,
        "count_functions_tokens",
        return_value=mock_count_functions_tokens_response,
    ):
        # Patch the count_chatml_tokens method in the GoogleTokenizer class
        with patch.object(
            GoogleTokenizer,
            "count_chatml_tokens",
            return_value=mock_count_chatml_tokens_response,
        ):
            tokenizer = GoogleTokenizer(model_name="gemini-1.0-pro-002")
            simple_chat: Chat = [
                {"content": "Debug bot.", "role": "system"},
                {
                    "content": "example return: ('Hello', 42)",
                    "name": "exampleFunction",
                    "role": "function",
                },
            ]
            functions = [example_function_jsonschema]
            assert tokenizer.count_functions_tokens(functions) == 23
            assert tokenizer.count_chatml_tokens(simple_chat, functions) == 39


def test_function_role_systemless(example_function_jsonschema: dict, mocker):
    # Create mock responses for count_functions_tokens and count_chatml_tokens
    mock_count_functions_tokens_response = 23
    mock_count_chatml_tokens_response = 36
    mocker.patch("totokenizers.google.GoogleTokenizer.load_gemini", authentication_mock)
    # Patch the count_functions_tokens method in the GoogleTokenizer class
    with patch.object(
        GoogleTokenizer,
        "count_functions_tokens",
        return_value=mock_count_functions_tokens_response,
    ):
        # Patch the count_chatml_tokens method in the GoogleTokenizer class
        with patch.object(
            GoogleTokenizer,
            "count_chatml_tokens",
            return_value=mock_count_chatml_tokens_response,
        ):
            tokenizer = GoogleTokenizer(model_name="gemini-1.0-pro-002")
            simple_chat: Chat = [
                {
                    "content": "example return: ('Hello', 42)",
                    "name": "exampleFunction",
                    "role": "function",
                },
            ]
            functions = [example_function_jsonschema]
            assert tokenizer.count_functions_tokens(functions) == 23
            assert tokenizer.count_chatml_tokens(simple_chat, functions) == 36


def test_functioncall_message(example_function_jsonschema: dict, mocker):
    # Create mock responses for count_functions_tokens and count_chatml_tokens
    mock_count_functions_tokens_response = 23
    mock_count_chatml_tokens_response = 35
    mocker.patch("totokenizers.google.GoogleTokenizer.load_gemini", authentication_mock)
    # Patch the count_functions_tokens method in the GoogleTokenizer class
    with patch.object(
        GoogleTokenizer,
        "count_functions_tokens",
        return_value=mock_count_functions_tokens_response,
    ):
        # Patch the count_chatml_tokens method in the GoogleTokenizer class
        with patch.object(
            GoogleTokenizer,
            "count_chatml_tokens",
            return_value=mock_count_chatml_tokens_response,
        ):
            tokenizer = GoogleTokenizer(model_name="gemini-1.0-pro-002")
            simple_chat = [
                {"content": "Debug bot.", "role": "system"},
                {
                    "content": None,
                    "function_call": {
                        "name": "exampleFunction",
                        "arguments": '{\n  "param1": "Hello",\n  "param2": 42\n}',
                    },
                    "role": "assistant",
                },
            ]
            functions = [example_function_jsonschema]
            assert tokenizer.count_functions_tokens(functions) == 23
            assert tokenizer.count_chatml_tokens(simple_chat, functions) == 35


def test_functioncall_message_systemless(example_function_jsonschema: dict, mocker):
    # Create mock responses for count_functions_tokens and count_chatml_tokens
    mock_count_functions_tokens_response = 23
    mock_count_chatml_tokens_response = 32
    mocker.patch("totokenizers.google.GoogleTokenizer.load_gemini", authentication_mock)
    # Patch the count_functions_tokens method in the GoogleTokenizer class
    with patch.object(
        GoogleTokenizer,
        "count_functions_tokens",
        return_value=mock_count_functions_tokens_response,
    ):
        # Patch the count_chatml_tokens method in the GoogleTokenizer class
        with patch.object(
            GoogleTokenizer,
            "count_chatml_tokens",
            return_value=mock_count_chatml_tokens_response,
        ):
            tokenizer = GoogleTokenizer(model_name="gemini-1.0-pro-002")
            simple_chat: Chat = [
                {
                    "content": None,
                    "function_call": {
                        "name": "exampleFunction",
                        "arguments": '{\n  "param1": "Hello",\n  "param2": 42\n}',
                    },
                    "role": "assistant",
                },
            ]
            functions = [example_function_jsonschema]
            assert tokenizer.count_functions_tokens(functions) == 23
            assert tokenizer.count_chatml_tokens(simple_chat, functions) == 32
