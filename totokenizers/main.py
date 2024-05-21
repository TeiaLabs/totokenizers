from .google import GoogleTokenizer
from .schemas import (
    ChatMLMessage,
    FunctionCallChatMLMessage,
    FunctionChatMLMessage,
    FunctionCall,
)
from vertexai.preview.generative_models import (
    GenerativeModel,
)
import vertexai

# from .test_inputs import *

if __name__ == "__main__":

    tokenizer = GoogleTokenizer("gemini-pro")
    vertexai.init(project="teia-cloud", location="us-central1")
    # Initialize Gemini model
    model = GenerativeModel(
        model_name="gemini-1.0-pro-002",
    )

    example = {
        "messages": [
            ChatMLMessage(
                **{"content": "What is the weather like in Boston?", "role": "user"}
            ),
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
                **{
                    "content": '{"temperature": "22", "unit": "celsius", "description": "Sunny"}',
                    "name": "get_current_weather",
                    "role": "function",
                }
            ),
        ],
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
        "settings": {"model": "gpt-3.5-turbo-0613"},
    }
    new_example = []
    # print(
    #     sum(
    #         [tokenizer.count_message_tokens(message) for message in example["messages"]]
    #     )
    # )
    # new_example.append(
    #     tokenizer._translate_chat_ml_message_to_gemini(example["messages"][0])
    # )
    # new_example.append(
    #     tokenizer._translate_function_call_chat_ml_message_to_gemini(
    #         example["messages"][1]
    #     )
    # )
    # new_example.append(
    #     tokenizer._translate_function_chat_ml_message_to_gemini(example["messages"][2])
    # )
    print(
        tokenizer.count_chatml_tokens(
            [
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
        )
    )

    print(
        model.generate_content(
            [
                tokenizer._translate_chat_ml_message_to_gemini(
                    ChatMLMessage(
                        content="hello bot",
                        name="user",
                        role="user",
                    )
                ),
                tokenizer._translate_chat_ml_message_to_gemini(
                    ChatMLMessage(
                        content="I am Skynet.",
                        name="skynet",
                        role="assistant",
                    )
                ),
            ]
        )
    )
    # print(model.generate_content(new_example))
    # É para 43
