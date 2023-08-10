
from .openai import OpenAITokenizer
from .errors import ModelNotFound


class Totokenizer:

    @classmethod
    def from_model(cls, model: str):
        try:
            provider, model_name = model.split("/", 1)
        except ValueError:
            raise ModelNotFound(model)
        if provider == "openai":
            return OpenAITokenizer(model_name)  # type: ignore
        raise ModelNotFound(model)
