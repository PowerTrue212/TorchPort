from __future__ import annotations

import json
import unittest

from experiments.harness.deepseek import DeepSeekAgent, DeepSeekConfig, _usage


class DeepSeekConfigTests(unittest.TestCase):
    def test_reads_environment_without_exposing_key(self) -> None:
        config = DeepSeekConfig.from_env(
            {
                "DEEPSEEK_API_KEY": "secret-key",
                "DEEPSEEK_MODEL": "deepseek-flash",
                "DEEPSEEK_MAX_MODEL_TURNS": "7",
            }
        )

        self.assertEqual(config.api_key, "secret-key")
        self.assertEqual(config.max_model_turns, 7)
        self.assertNotIn("secret-key", json.dumps(config.public()))

    def test_placeholder_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
            DeepSeekConfig.from_env({"DEEPSEEK_API_KEY": "replace_me"})


class DeepSeekAgentTests(unittest.TestCase):
    def test_runs_tool_loop_and_records_provider_usage(self) -> None:
        requests = []
        events = []
        replies = iter(
            [
                {
                    "id": "r1",
                    "model": "deepseek-flash",
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "reasoning_content": "I should inspect the task first.",
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "type": "function",
                                        "function": {"name": "inspect", "arguments": "{}"},
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 4,
                        "total_tokens": 14,
                        "prompt_tokens_details": {"cached_tokens": 3},
                        "completion_tokens_details": {"reasoning_tokens": 0},
                    },
                },
                {
                    "id": "r2",
                    "model": "deepseek-flash",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "done"},
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
                },
            ]
        )

        def transport(url, headers, body, timeout):
            requests.append((url, headers, body, timeout))
            return next(replies)

        calls = []

        def call(name, arguments):
            calls.append((name, arguments))
            return {"ok": True, "data": {}, "budget": {"budget_stop_reason": None}}

        agent = DeepSeekAgent(
            DeepSeekConfig(api_key="secret"),
            "migrate",
            recorder=lambda event, **facts: events.append((event, facts)),
            transport=transport,
        )
        reason = agent(
            [{"name": "inspect", "description": "inspect", "parameters": {"type": "object"}}],
            call,
        )

        self.assertEqual(reason, "agent_completed")
        self.assertEqual(calls, [("inspect", {})])
        self.assertEqual(requests[0][0], "https://api.deepseek.com/chat/completions")
        self.assertEqual(requests[0][2]["thinking"], {"type": "enabled"})
        self.assertEqual(requests[0][2]["reasoning_effort"], "max")
        self.assertNotIn("temperature", requests[0][2])
        self.assertEqual(events[0][1]["usage"]["cached_input_tokens"], 3)
        assistant_message = requests[1][2]["messages"][-2]
        self.assertEqual(
            assistant_message["reasoning_content"], "I should inspect the task first."
        )
        tool_message = requests[1][2]["messages"][-1]
        self.assertEqual(tool_message["role"], "tool")
        self.assertNotIn("secret", json.dumps(events))

    def test_budget_stops_without_another_request(self) -> None:
        requested = []
        agent = DeepSeekAgent(
            DeepSeekConfig(api_key="secret"),
            "migrate",
            budget=lambda: {
                "budget_stop_reason": "oracle_query_budget_exhausted",
                "wall_seconds_remaining": 10,
            },
            transport=lambda *_args: requested.append(True),
        )

        reason = agent([], lambda _name, _arguments: {})

        self.assertEqual(reason, "oracle_query_budget_exhausted")
        self.assertEqual(requested, [])

    def test_total_token_budget_keeps_last_response_tool_call(self) -> None:
        calls = []
        response = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "inspect", "arguments": "{}"},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
        }
        agent = DeepSeekAgent(
            DeepSeekConfig(api_key="secret"),
            "migrate",
            max_total_tokens=10,
            transport=lambda *_args: response,
        )

        def call(name, arguments):
            calls.append((name, arguments))
            return {"budget": {"budget_stop_reason": None}}

        reason = agent([], call)

        self.assertEqual(reason, "token_budget_exhausted")
        self.assertEqual(calls, [("inspect", {})])

    def test_usage_preserves_raw_token_classes(self) -> None:
        self.assertEqual(
            _usage(
                {
                    "prompt_tokens": 9,
                    "completion_tokens": 4,
                    "total_tokens": 13,
                    "prompt_cache_hit_tokens": 2,
                    "completion_tokens_details": {"reasoning_tokens": 1},
                }
            ),
            {
                "input_tokens": 9,
                "cached_input_tokens": 2,
                "output_tokens": 4,
                "reasoning_tokens": 1,
                "total_tokens": 13,
            },
        )


if __name__ == "__main__":
    unittest.main()
