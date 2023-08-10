
from .openai import OpenAITokenizer
from .errors import ModelNotFound, ModelProviderNotFound, BadFormatForModelTag
from .openai_info import OPEN_AI_MODELS


class Totokenizer:

    @classmethod
    def from_model(cls, model: str):
        try:
            provider, model_name = model.split("/", 1)
        except (ValueError, TypeError):
            raise BadFormatForModelTag(model)
        if provider == "openai":
            return OpenAITokenizer(model_name)  # type: ignore
        raise ModelProviderNotFound(provider)


class TotoModelInfo:

    @classmethod
    def from_model(cls, model: str):
        try:
            provider, model_name = model.split("/", 1)
        except ValueError:
            raise BadFormatForModelTag(model)
        if provider == "openai":
            if model_name not in OPEN_AI_MODELS:
                raise ModelNotFound(model_name)
            return OPEN_AI_MODELS[model_name]
        raise ModelProviderNotFound(provider)
