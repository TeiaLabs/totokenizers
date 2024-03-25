import json
from typing import List

from pathlib import Path
from typing import Literal, Sequence

from transformers import GemmaTokenizer
from tokenizers import (
    Encoding,
    Tokenizer as HFTokenizer,
)

from ..schemas import ChatMLMessage    


class Gemma:
    """
    Tokenizer for the Google Gemma models.

    Args:
        model_name (str): The name of the model to use.


    The tokenizer is based on HuggingFace's `tokenizers` package:
    https://github.com/huggingface/tokenizers
    """
    def __init__(
        self,
        model_name: Literal[
            "Gemma-2b",
            "Gemma-2b-it",
            "Gemma-7b",
            "Gemma-7b-it"
        ],
    ):
        chat_template = json.load(open(f"totokenizers/google/{model_name}/tokenizer_config.json")).get("chat_template", None)
        self.tokenizer_path = Path(__file__).parent / "tokenizer.json"
        self.encoder: HFTokenizer = GemmaTokenizer(vocab_file=f"totokenizers/google/{model_name}/tokenizer.model",
                                                   chat_template=chat_template, 
                                                   **json.load(open(f"totokenizers/google/{model_name}/tokenizer.json")))
        self.model_name = model_name

    def encode(self, text: str, add_special_tokens:bool=True) -> list[int]:
        encoded: Encoding = self.encoder.encode(text, add_special_tokens=add_special_tokens)
        return encoded
    
    def decode(self, encoded_text:List[int]):

        return self.encoder.decode(encoded_text)
    
    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text.
        """
        return len(self.encode(text, add_special_tokens=False))

    def count_chatml_message_tokens(self, message: ChatMLMessage) -> int:
        """
        Counts the number of tokens in a given ChatMLMessage
        """
        raw_message = self.encoder.apply_chat_template(conversation=[message])
        return len(raw_message)

    def count_chatml_tokens(self, messages: Sequence[ChatMLMessage]) -> int:
        """
        Counts the number of tokens in a given ChatMLMessage list
        """
        num_tokens = len(self.encoder.apply_chat_template(conversation=messages, add_special_tokens=True))
        return num_tokens

    def count_chatml_prompt_tokens(self, messages: Sequence[ChatMLMessage]) -> int:
        """
        The aply_chat_template generates the exactly number of tokens
        that was expected including the new tokens 
        """
        num_tokens = self.count_chatml_tokens(messages)
        return num_tokens

    def count_completion_tokens(self, text: str) -> int:
        """
        Returns a count that matches the "completion tokens" in the logs.
        """
        # A completion response always starts with "\n\nassistant:"
        num_tokens = self.count_tokens(text)
        return num_tokens
