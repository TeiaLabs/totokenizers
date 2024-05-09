"""
https://github.com/Significant-Gravitas/Auto-GPT/blob/3a2d08fb415071cc94dd6fcee24cfbdd1fb487dd/autogpt/llm/base.py#L47
"""

from .model_info import ChatModelInfo, TextModelInfo, EmbeddingModelInfo


OPEN_AI_CHAT_MODELS = {
    info.name: info
    for info in [
        ChatModelInfo(
             completion_token_cost=0.03,
             cutoff="2023-12-01",
             max_tokens=128_000,
             name="gpt-4-turbo-2024-04-09",
             prompt_token_cost=0.01,
             supports_functions=True,
         ),
        ChatModelInfo(
            completion_token_cost=0.03,
            cutoff="2023-04-01",
            max_tokens=128000,
            name="gpt-4-0125-preview",
            prompt_token_cost=0.01,
            supports_functions=True,
        ),
        ChatModelInfo(
            completion_token_cost=0.03,
            cutoff="2023-04-01",
            max_tokens=128000,
            name="gpt-4-turbo-preview",
            prompt_token_cost=0.01,
            supports_functions=True,
        ),
        ChatModelInfo(
            completion_token_cost=0.03,
            cutoff="2023-04-01",
            max_tokens=128000,
            name="gpt-4-1106-preview",
            prompt_token_cost=0.01,
            supports_functions=True,
        ),
        ChatModelInfo(
            completion_token_cost=0.03,
            cutoff="2023-04-01",
            max_tokens=128000,
            name="gpt-4-1106-vision-preview",
            prompt_token_cost=0.01,
            supports_functions=True,
        ),
        ChatModelInfo(
            completion_token_cost=0.0015,
            cutoff="2021-09-01",
            max_tokens=16385,
            name="gpt-3.5-turbo-0125",
            prompt_token_cost=0.0005,
            supports_functions=True,
        ),
        ChatModelInfo(
            completion_token_cost=0.002,
            cutoff="2021-11-01",
            max_tokens=16385,
            name="gpt-3.5-turbo-1106",
            prompt_token_cost=0.001,
            supports_functions=True,
        ),
        ChatModelInfo(
            completion_token_cost=0.002,
            cutoff="2021-11-01",
            max_tokens=4096,
            name="gpt-3.5-turbo-0301",
            prompt_token_cost=0.0015,
        ),
        ChatModelInfo(
            completion_token_cost=0.002,
            cutoff="2021-11-01",
            max_tokens=4096,
            name="gpt-3.5-turbo-0613",
            prompt_token_cost=0.0015,
            supports_functions=True,
        ),
        ChatModelInfo(
            completion_token_cost=0.004,
            cutoff="2021-11-01",
            max_tokens=16385,
            name="gpt-3.5-turbo-16k-0613",
            prompt_token_cost=0.003,
            supports_functions=True,
        ),
        ChatModelInfo(
            completion_token_cost=0.06,
            cutoff="2021-11-01",
            max_tokens=8192,
            name="gpt-4-0314",
            prompt_token_cost=0.03,
        ),
        ChatModelInfo(
            completion_token_cost=0.06,
            cutoff="2021-11-01",
            max_tokens=8192,
            name="gpt-4-0613",
            prompt_token_cost=0.03,
            supports_functions=True,
        ),
        ChatModelInfo(
            completion_token_cost=0.12,
            cutoff="2021-11-01",
            max_tokens=32768,
            name="gpt-4-32k-0314",
            prompt_token_cost=0.06,
        ),
        ChatModelInfo(
            completion_token_cost=0.12,
            cutoff="2021-11-01",
            max_tokens=32768,
            name="gpt-4-32k-0613",
            prompt_token_cost=0.06,
            supports_functions=True,
        ),
    ]
}

# Set aliases for rolling model IDs
chat_model_mapping = {
    "gpt-3.5-turbo": "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k": "gpt-3.5-turbo-16k-0613",
    "gpt-4": "gpt-4-0613",
    "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
    "gpt-4-32k": "gpt-4-32k-0613",
}
for alias, target in chat_model_mapping.items():
    alias_info = ChatModelInfo(**OPEN_AI_CHAT_MODELS[target].__dict__)
    alias_info.name = alias
    OPEN_AI_CHAT_MODELS[alias] = alias_info

OPEN_AI_TEXT_MODELS = {
    info.name: info
    for info in [
        TextModelInfo(
            completion_token_cost=0.02,
            cutoff="2021-11-01",
            max_tokens=4096,
            name="text-davinci-003",
            prompt_token_cost=0.02,
        ),
        TextModelInfo(
            completion_token_cost=0.002,
            cutoff="2021-11-01",
            max_tokens=4096,
            name="gpt-3.5-turbo-instruct",
            prompt_token_cost=0.0015,
        ),
    ]
}

OPEN_AI_EMBEDDING_MODELS = {
    info.name: info
    for info in [
        EmbeddingModelInfo(
            cutoff="2021-11-01",
            embedding_dimensions=1536,
            max_tokens=8191,
            name="text-embedding-ada-002",
            prompt_token_cost=0.0001,
        ),
    ]
}

OPEN_AI_MODELS: dict[str, ChatModelInfo | EmbeddingModelInfo | TextModelInfo] = {
    **OPEN_AI_CHAT_MODELS,
    **OPEN_AI_TEXT_MODELS,
    **OPEN_AI_EMBEDDING_MODELS,
}
