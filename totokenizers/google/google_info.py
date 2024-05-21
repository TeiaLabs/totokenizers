from ..model_info import ChatModelInfo, EmbeddingModelInfo, TextModelInfo

GOOGLE_CHAT_MODELS = {
    info.name: info
    for info in [
        ChatModelInfo(
            completion_token_cost=0.015,
            cutoff="2024-02-01",
            feature_flags=["functions", "tools", "json"],
            max_output_tokens=8_192,
            max_tokens=24_568,
            name="gemini-1.0-pro-latest",
            prompt_token_cost=0.005,
        ),
        ChatModelInfo(
            completion_token_cost=0.015,
            cutoff="2024-02-01",
            feature_flags=["functions", "tools", "json"],
            max_output_tokens=8_192,
            max_tokens=24_568,
            name="gemini-1.0-pro",
            prompt_token_cost=0.005,
        ),
        ChatModelInfo(
            completion_token_cost=0.015,
            cutoff="2023-12-01",
            feature_flags=["vision", "voice"],
            max_output_tokens=2_048,
            max_tokens=16_384,
            name="gemini-1.0-pro-vision-latest",
            prompt_token_cost=0.005,
        ),
        ChatModelInfo(
            completion_token_cost=0.015,
            cutoff="2023-12-01",
            feature_flags=["vision", "voice"],
            max_output_tokens=2_048,
            max_tokens=16_384,
            name="gemini-1.0-pro-vision",
            prompt_token_cost=0.005,
        ),
    ]
}

GOOGLE_MODELS: dict[str, ChatModelInfo] = {
    **GOOGLE_CHAT_MODELS,
}
