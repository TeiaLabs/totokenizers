from ..model_info import ChatModelInfo


CHAT_MODELS = {
    info.name: info
    for info in [
        ChatModelInfo(
            completion_token_cost=0,
            cutoff="1997-01-01",
            max_tokens=4096,
            name="always-func",
            prompt_token_cost=0,
        ),
        ChatModelInfo(
            completion_token_cost=0,
            cutoff="1997-01-01",
            max_tokens=4096,
            name="always-chat",
            prompt_token_cost=0,
            supports_functions=True,
        ),
    ]
}

MODELS: dict[str, ChatModelInfo] = {**CHAT_MODELS}
