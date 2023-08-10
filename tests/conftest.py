import pytest


@pytest.fixture(scope="session")
def example_function_jsonschema():
    d = {
        "name": "exampleFunction",
        "description": "This is an example function",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"description": "This is parameter 1", "type": "string"},
                "param2": {
                    "description": "This is parameter 2",
                    "type": "number",
                    "minimum": 0,
                    "maximum": 10,
                },
            },
            "required": ["param1"],
        },
    }
    return d


@pytest.fixture(scope="session")
def example_function2_jsonschema():
    d = {
        "name": "function_name",
        "description": "Description of example function the AI will repeat back to the user",
        "parameters": {
            "type": "object",
            "properties": {
                "property1": {
                    "type": "string",
                    "description": "description of function property 1: string",
                },
                "property2": {
                    "type": "string",
                    "enum": ["enum_yes", "enum_no"],
                    "description": "description of function property 2: string w enum",
                },
            },
            "required": ["required_properties"],
        },
    }
    return d
