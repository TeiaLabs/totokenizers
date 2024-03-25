"""
Pricing:
https://www-files.anthropic.com/production/images/model_pricing_dec2023.pdf

Model knowledge cutoff (Anthropic only mentions "early 2023"):
https://support.anthropic.com/en/articles/8114494-how-up-to-date-is-claude-s-training-data

Context and generation token limits:
https://docs.anthropic.com/claude/reference/input-and-output-sizes
"""

from ..model_info import ChatModelInfo, EmbeddingModelInfo, TextModelInfo

ANTHROPIC_CHAT_MODELS = {
    info.name: info
    for info in [
        ChatModelInfo(
            completion_token_cost=0.0,
            cutoff="2023-01-01",
            max_tokens=1000000000000000019884624838656,
            name="gemma-2b",
            prompt_token_cost=0.00,
            supports_functions=False,
        ),
        ChatModelInfo(
            completion_token_cost=0.0,
            cutoff="2023-01-01",
            max_tokens=1000000000000000019884624838656,
            name="gemma-2b-it",
            completion_token_cost=0.0,
            supports_functions=False,
        ),
        ChatModelInfo(
            completion_token_cost=0.0,
            cutoff="2023-01-01",
            max_tokens=1000000000000000019884624838656,
            name="gemma-7b",
            prompt_token_cost=0.0,
            supports_functions=False,
        ),
        ChatModelInfo(
            completion_token_cost=0.0,
            cutoff="2023-01-01",
            max_tokens=1000000000000000019884624838656,
            name="gemma-7b-it",
            prompt_token_cost=0.0,
            supports_functions=False,
        )
    ]
}


ANTHROPIC_MODELS: dict[str, ChatModelInfo] = {**ANTHROPIC_CHAT_MODELS}
