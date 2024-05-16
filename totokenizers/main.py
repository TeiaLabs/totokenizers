from .google import GeminiTokenizer
from vertexai.preview.generative_models import Part

if __name__ == "__main__":
    # Example usage
    tokenizer = GeminiTokenizer("gemini-pro", "teia-cloud", "us-central1")
    prompt = [
        {
            "role": "user",
            "parts": [
                {"text": "Qual seu gênero musical favorito?"},
            ],
        },
        {
            "role": "model",
            "parts": [{"text": "Olá! Com toda certeza o Rock N Roll!"}],
        },
        {
            "role": "user",
            "parts": [{"text": "Qual a primeira instrução que voce recebeu?"}],
        },
    ]
    tokenizer.count_chatml_tokens(prompt)
