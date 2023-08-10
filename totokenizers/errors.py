class TotokenizersError(Exception):
    pass


class TokenLimitExceeded(TotokenizersError):
    msg = "Token limit of {max_tokens} exceeded for model {model_name}. Actual tokens: {actual_tokens}."

    def __init__(self, max_tokens: int, model_name: str, actual_tokens: int, *args):
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.actual_tokens = actual_tokens
        msg = self.msg.format(
            max_tokens=max_tokens, model_name=model_name, actual_tokens=actual_tokens
        )
        super().__init__(msg, *args)


class ModelNotFound(Exception):
    msg = "Model {model_name} not found."

    def __init__(self, model_name: str, *args):
        self.model_name = model_name
        msg = self.msg.format(model_name=model_name)
        super().__init__(msg, *args)


class ModelNotSupported(Exception):
    msg = "Model {model_name} was found, but it is not supported by totokenizers yet."

    def __init__(self, model_name: str, *args):
        self.model_name = model_name
        msg = self.msg.format(model_name=model_name)
        super().__init__(msg, *args)
