# Google Gemini Tokenizer

Google did not publicly release their tokenizer for Gemini models, and there is little to no information about it online.
The model's prompt format is also unclear at the moment.
We have two options to get access to token counts:

- Use the Vertex AI Python SDK, or
- Call the Vertex AI REST API directly.

[Reference for token counting via SDK/API](https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/get-token-count)

For both options, we still have to set up [Google Cloud authentication](https://googleapis.dev/python/google-api-core/latest/auth.html) beforehand, since the SDK searches for keys during its initialization and the API requires us to send them as well.
There are several ways to authenticate with Google Cloud, and this should probably not be done by totokenizers.
With that said, if we choose to use the API directly, we can request that users pass the authentication token during the tokenizer initialization, or we can use the [`google-auth`](https://google-auth.readthedocs.io/en/master/) library to get the necessary keys (probably via the `default()` method, which tries to find the keys in the environment, A.K.A. the [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials)).
This package has less dependencies than using the Python SDK, which requires us to install the [`google-cloud-aiplatform`](https://cloud.google.com/python/docs/reference/aiplatform/latest) package (i.e., bloatware).
However, I believe that it is possible that the SDK returns more information about tokens, such as billing information (see [`CountTokensResponse`](https://github.com/googleapis/python-aiplatform/blob/main/google/cloud/aiplatform_v1beta1/types/prediction_service.py) and [`count_tokens`](https://github.com/googleapis/python-aiplatform/blob/1fbf0493dc5fa2bb05f33a4319d79a81625e07cc/vertexai/generative_models/_generative_models.py)).
We would have to investigate this further.
