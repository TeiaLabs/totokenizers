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
            max_output_tokens=-1,
        ),
        ChatModelInfo(
            completion_token_cost=0,
            cutoff="1997-01-01",
            max_tokens=4096,
            name="always-chat",
            prompt_token_cost=0,
            feature_flags=True,
            max_output_tokens=-1,
        ),
    ]
}

MODELS: dict[str, ChatModelInfo] = {**CHAT_MODELS}
