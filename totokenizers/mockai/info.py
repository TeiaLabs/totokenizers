from ..model_info import ChatModelInfo


CHAT_MODELS = {
    info.name: info
    for info in [
        ChatModelInfo(
            name="always-func",
            prompt_token_cost=0,
            completion_token_cost=0,
            max_tokens=4096,
        ),
        ChatModelInfo(
            name="always-chat",
            prompt_token_cost=0,
            completion_token_cost=0,
            max_tokens=4096,
            supports_functions=True,
        ),
    ]
}

MODELS: dict[str, ChatModelInfo] = {**CHAT_MODELS}
