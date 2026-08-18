import pytest
import requests
from unittest.mock import MagicMock, patch
from model_adapter import infer, get_adapter
from model_adapter.base import BaseAdapter

def test_interface_compliance():
    """Verify that adapters implement BaseAdapter and return expected structure keys."""
    class DummyAdapter(BaseAdapter):
        def infer(self, messages, tools=None, config=None):
            return {
                "text": "Hello world",
                "tool_calls": [],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                "stop_reason": "stop"
            }

    adapter = DummyAdapter()
    result = adapter.infer([{"role": "user", "content": "hi"}])
    
    assert isinstance(result, dict)
    assert "text" in result
    assert "tool_calls" in result
    assert "usage" in result
    assert "stop_reason" in result
    assert result["text"] == "Hello world"
    assert result["usage"]["total_tokens"] == 7

@patch("model_adapter.providers.openrouter.requests.post")
def test_openrouter_adapter_infer(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Test response",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Tokyo"}'
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25
        }
    }
    mock_post.return_value = mock_response

    result = infer(
        messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
        config={"provider": "openrouter", "model_id": "openrouter/free"}
    )

    assert result["text"] == "Test response"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["name"] == "get_weather"
    assert result["tool_calls"][0]["arguments"] == {"location": "Tokyo"}
    assert result["usage"]["total_tokens"] == 25
    assert result["stop_reason"] == "tool_calls"

@patch("model_adapter.providers.openrouter.requests.post")
def test_infer_retry_logic(mock_post):
    # First two calls raise ConnectionError, third call succeeds
    mock_post.side_effect = [
        requests.exceptions.ConnectionError("Connection aborted"),
        requests.exceptions.Timeout("Read timed out"),
        MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": "Success after retry"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
            }
        )
    ]

    result = infer(
        messages=[{"role": "user", "content": "Hello"}],
        config={"provider": "openrouter", "model_id": "openrouter/free"}
    )

    assert result["text"] == "Success after retry"
    assert mock_post.call_count == 3

@patch("model_adapter.providers.openrouter.requests.post")
def test_infer_no_retry_on_client_error(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "Bad Request"
    mock_post.return_value = mock_response

    with pytest.raises(RuntimeError, match="OpenRouter API error \\(400\\)"):
        infer(
            messages=[{"role": "user", "content": "Hello"}],
            config={"provider": "openrouter", "model_id": "openrouter/free"}
        )

    # Ensure it did not retry (called only once)
    assert mock_post.call_count == 1

@patch("model_adapter.providers.openrouter.requests.post")
def test_infer_retry_on_rate_limit_429(mock_post):
    mock_429 = MagicMock()
    mock_429.status_code = 429
    mock_429.text = "Rate limit exceeded"

    mock_success = MagicMock()
    mock_success.status_code = 200
    mock_success.json.return_value = {
        "choices": [{"message": {"content": "Success after 429 retry"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
    }

    mock_post.side_effect = [mock_429, mock_success]

    result = infer(
        messages=[{"role": "user", "content": "Hello"}],
        config={"provider": "openrouter", "model_id": "openrouter/free"}
    )

    assert result["text"] == "Success after 429 retry"
    assert mock_post.call_count == 2

@patch("model_adapter.providers.openrouter.requests.post")
def test_infer_retry_on_server_error_503(mock_post):
    mock_503 = MagicMock()
    mock_503.status_code = 503
    mock_503.text = "Service Unavailable"

    mock_post.side_effect = [mock_503, mock_503, mock_503]

    with pytest.raises(RuntimeError, match="Model inference failed after 3 attempts"):
        infer(
            messages=[{"role": "user", "content": "Hello"}],
            config={"provider": "openrouter", "model_id": "openrouter/free"}
        )

    assert mock_post.call_count == 3

@patch("model_adapter.providers.openrouter.requests.post")
def test_tool_call_json_string_arguments_parsing(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "search_database",
                                "arguments": '{"query": "machine learning", "limit": 10}'
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "total_tokens": 20
        }
    }
    mock_post.return_value = mock_response

    result = infer(
        messages=[{"role": "user", "content": "Search for machine learning"}],
        tools=[{"type": "function", "function": {"name": "search_database"}}],
        config={"provider": "openrouter", "model_id": "openrouter/free"}
    )

    assert len(result["tool_calls"]) == 1
    tc = result["tool_calls"][0]
    assert tc["id"] == "call_abc123"
    assert tc["name"] == "search_database"
    assert isinstance(tc["arguments"], dict)
    assert tc["arguments"]["query"] == "machine learning"
    assert tc["arguments"]["limit"] == 10

@patch("model_adapter.providers.openrouter.requests.post")
def test_infer_missing_usage_fields(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"content": "Response without usage"},
                "finish_reason": "stop"
            }
        ]
        # 'usage' key is completely missing
    }
    mock_post.return_value = mock_response

    result = infer(
        messages=[{"role": "user", "content": "Test"}],
        config={"provider": "openrouter", "model_id": "openrouter/free"}
    )

    assert result["text"] == "Response without usage"
    assert result["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }

@patch("model_adapter.providers.openrouter.requests.post")
def test_infer_alternative_usage_keys(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"content": "Response with alt usage keys"},
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "input_tokens": 42,
            "output_tokens": 18
        }
    }
    mock_post.return_value = mock_response

    result = infer(
        messages=[{"role": "user", "content": "Test alt usage"}],
        config={"provider": "openrouter", "model_id": "openrouter/free"}
    )

    assert result["text"] == "Response with alt usage keys"
    assert result["usage"] == {
        "prompt_tokens": 42,
        "completion_tokens": 18,
        "total_tokens": 60
    }

@patch("model_adapter.providers.openrouter.requests.post")
def test_infer_custom_stop_reasons(mock_post):
    for finish_reason in ["length", "tool_calls", "content_filter"]:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {"content": "Stop reason test"},
                    "finish_reason": finish_reason
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        }
        mock_post.return_value = mock_response

        result = infer(
            messages=[{"role": "user", "content": "Test stop reason"}],
            config={"provider": "openrouter", "model_id": "openrouter/free"}
        )

        assert result["stop_reason"] == finish_reason

@patch("model_adapter.providers.openrouter.requests.post")
def test_infer_model_tiers_routing(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Tier response"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
    }
    mock_post.return_value = mock_response

    # Test planner tier
    result_planner = infer(
        messages=[{"role": "user", "content": "Plan step"}],
        config={"provider": "openrouter", "model_id": "openrouter/planner-model"}
    )
    assert result_planner["text"] == "Tier response"
    
    # Verify model parameter sent in request payload
    called_payload = mock_post.call_args[1]["json"]
    assert called_payload["model"] == "planner-model"

    # Test executor tier
    result_executor = infer(
        messages=[{"role": "user", "content": "Execute step"}],
        config={"provider": "openrouter", "model_id": "openrouter/executor-model"}
    )
    assert result_executor["text"] == "Tier response"

    called_payload_exec = mock_post.call_args[1]["json"]
    assert called_payload_exec["model"] == "executor-model"

@patch("model_adapter.providers.openrouter.requests.post")
def test_infer_model_config_dict_tiers(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Dict tier response"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
    }
    mock_post.return_value = mock_response

    # Call with model config dictionary specifying planner and executor
    config = {
        "provider": "openrouter",
        "model": {
            "planner": "openrouter/gpt-4o",
            "executor": "openrouter/claude-3-5-sonnet"
        },
        "tier": "executor"
    }

    result = infer(
        messages=[{"role": "user", "content": "Execute task"}],
        config=config
    )

    assert result["text"] == "Dict tier response"
    called_payload = mock_post.call_args[1]["json"]
    assert called_payload["model"] == "claude-3-5-sonnet"
