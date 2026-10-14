import http.client
import base64
import json
import threading
import unittest
from unittest import mock
from urllib.parse import parse_qs

import importlib.util
from pathlib import Path
from gemini_web2api.config import CONFIG, DEFAULT_CONFIG
from gemini_web2api.gemini import _build_headers, _build_payload
from gemini_web2api.models import MODELS, build_model_ticket, resolve_model
from gemini_web2api.server import GeminiHandler, ThreadedServer
from gemini_web2api.tools import google_contents_to_prompt, messages_to_prompt, parse_tool_calls

_SINGLE_SPEC = importlib.util.spec_from_file_location(
    "gemini_web2api_single",
    Path(__file__).resolve().parent.parent / "gemini_web2api.py",
)
single_file = importlib.util.module_from_spec(_SINGLE_SPEC)
_SINGLE_SPEC.loader.exec_module(single_file)


def _decode_payload(payload):
    outer = json.loads(parse_qs(payload)["f.req"][0])
    return json.loads(outer[1])


def _decode_sse(body):
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        event_type = next(
            (line[len("event: "):] for line in lines if line.startswith("event: ")),
            None,
        )
        data = next(
            (line[len("data: "):] for line in lines if line.startswith("data: ")),
            None,
        )
        if event_type and data:
            events.append((event_type, json.loads(data)))
    return events


class PayloadPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.original_config = dict(CONFIG)

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)

    def test_temporary_chats_default_to_disabled(self):
        self.assertIs(DEFAULT_CONFIG["temporary_chats"], False)

    def test_persistent_chat_payload(self):
        CONFIG["temporary_chats"] = False

        inner = _decode_payload(_build_payload("hello", 1, 4))

        self.assertEqual(inner[41], [2])
        self.assertIsNone(inner[45])

    def test_temporary_chat_payload(self):
        CONFIG["temporary_chats"] = True

        inner = _decode_payload(_build_payload("hello", 1, 4))

        self.assertEqual(inner[41], [1])
        self.assertEqual(inner[45], 1)

    def test_payload_includes_uploaded_image_refs(self):
        inner = _decode_payload(_build_payload("describe", 1, 4, ["/uploaded/image-ref"]))

        self.assertEqual(inner[0][0], "describe")
        self.assertEqual(inner[0][3], [[None, None, "/uploaded/image-ref"]])


class MessageParsingTests(unittest.TestCase):
    def test_messages_to_prompt_extracts_openai_image_url_data_url(self):
        image_data = base64.b64encode(b"fake png").decode()

        prompt, images = messages_to_prompt([{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
            ],
        }])

        self.assertEqual(prompt, "Describe [Image attached]")
        self.assertEqual(images, [(b"fake png", "image/png")])

    def test_messages_to_prompt_extracts_responses_input_image_url(self):
        prompt, images = messages_to_prompt([{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe"},
                {"type": "input_image", "image_url": "https://example.com/image.png"},
            ],
        }])

        self.assertEqual(prompt, "Describe [Image attached]")
        self.assertEqual(images, [("https://example.com/image.png", "image/png")])

    def test_messages_to_prompt_ignores_malformed_image_data_url(self):
        prompt, images = messages_to_prompt([{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,%%%"}},
            ],
        }])

        self.assertEqual(prompt, "Describe")
        self.assertEqual(images, [])

    def test_google_contents_to_prompt_extracts_inline_image_data(self):
        image_data = base64.b64encode(b"fake png").decode()

        prompt, images = google_contents_to_prompt({
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": "Describe"},
                    {"inlineData": {"mimeType": "image/png", "data": image_data}},
                ],
            }],
        })

        self.assertEqual(prompt, "Describe\n[Image attached]")
        self.assertEqual(images, [(b"fake png", "image/png")])

    def test_google_contents_to_prompt_ignores_malformed_inline_image_data(self):
        prompt, images = google_contents_to_prompt({
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": "Describe"},
                    {"inlineData": {"mimeType": "image/png", "data": "%%%"}},
                ],
            }],
        })

        self.assertEqual(prompt, "Describe")
        self.assertEqual(images, [])


class StreamingEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadedServer(("127.0.0.1", 0), GeminiHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        self.original_config = dict(CONFIG)
        CONFIG["api_keys"] = []
        CONFIG["log_requests"] = False

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)

    def post_json(self, path, payload):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request(
            "POST",
            path,
            body=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        body = response.read().decode()
        headers = dict(response.getheaders())
        connection.close()
        return response.status, headers, body

    def post_chunked_json(self, path, payload):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request(
            "POST",
            path,
            body=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            encode_chunked=True,
        )
        response = connection.getresponse()
        body = response.read().decode()
        headers = dict(response.getheaders())
        connection.close()
        return response.status, headers, body

    @mock.patch("gemini_web2api.server.generate_stream")
    def test_chat_stream_starts_with_assistant_role(self, generate_stream):
        generate_stream.return_value = iter(["hel", "lo"])

        status, headers, body = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        chunks = [
            json.loads(line[len("data: "):])
            for line in body.splitlines()
            if line.startswith("data: {")
        ]
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertEqual(chunks[1]["choices"][0]["delta"], {"content": "hel"})
        self.assertEqual(chunks[2]["choices"][0]["delta"], {"content": "lo"})
        self.assertTrue(body.endswith("data: [DONE]\n\n"))

    @mock.patch("gemini_web2api.server.generate", return_value="chunked ok")
    def test_chat_accepts_chunked_body(self, _generate):
        status, _, body = self.post_chunked_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "chunked ok")

    @mock.patch("gemini_web2api.server.upload_image", return_value="/uploaded/image-ref")
    @mock.patch("gemini_web2api.server.generate", return_value="looks good")
    def test_chat_accepts_openai_image_url_data_url(self, generate, upload_image):
        image_data = base64.b64encode(b"fake png").decode()

        status, _, body = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-3.6-flash",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            },
                        },
                    ],
                }],
            },
        )

        self.assertEqual(status, 200)
        upload_image.assert_called_once_with(b"fake png", "image.png", "image/png")
        self.assertEqual(generate.call_args.args[3], ["/uploaded/image-ref"])
        self.assertIn("[Image attached]", generate.call_args.args[0])
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "looks good")

    @mock.patch("gemini_web2api.server.fetch_image_bytes", return_value=b"\xff\xd8\xffremote jpeg")
    @mock.patch("gemini_web2api.server.upload_image", return_value="/uploaded/remote-ref")
    @mock.patch("gemini_web2api.server.generate", return_value="remote ok")
    def test_responses_accepts_input_image_url(self, generate, upload_image, fetch_image_bytes):
        status, _, _ = self.post_json(
            "/v1/responses",
            {
                "model": "gemini-3.6-flash",
                "input": [{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What is shown?"},
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/image.jpg",
                        },
                    ],
                }],
            },
        )

        self.assertEqual(status, 200)
        fetch_image_bytes.assert_called_once_with("https://example.com/image.jpg")
        upload_image.assert_called_once_with(b"\xff\xd8\xffremote jpeg", "image.png", "image/jpeg")
        self.assertEqual(generate.call_args.args[3], ["/uploaded/remote-ref"])
        self.assertIn("[Image attached]", generate.call_args.args[0])

    @mock.patch("gemini_web2api.server.upload_image", return_value="/uploaded/image-ref")
    @mock.patch("gemini_web2api.server.generate", return_value="top-level image ok")
    def test_responses_accepts_top_level_input_image(self, generate, upload_image):
        image_data = base64.b64encode(b"fake png").decode()

        status, _, _ = self.post_json(
            "/v1/responses",
            {
                "model": "gemini-3.6-flash",
                "input": [
                    {"type": "input_text", "text": "What is shown?"},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_data}",
                    },
                ],
            },
        )

        self.assertEqual(status, 200)
        upload_image.assert_called_once_with(b"fake png", "image.png", "image/png")
        self.assertEqual(generate.call_args.args[3], ["/uploaded/image-ref"])
        self.assertIn("What is shown?", generate.call_args.args[0])
        self.assertIn("[Image attached]", generate.call_args.args[0])

    @mock.patch("gemini_web2api.server.upload_image", side_effect=RuntimeError("upload denied"))
    def test_google_image_upload_failure_returns_502(self, _upload_image):
        image_data = base64.b64encode(b"fake png").decode()

        status, _, body = self.post_json(
            "/v1beta/models/gemini-3.6-flash:generateContent",
            {
                "contents": [{
                    "role": "user",
                    "parts": [{
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": image_data,
                        },
                    }],
                }],
            },
        )

        self.assertEqual(status, 502)
        self.assertIn("image upload failed: upload denied", json.loads(body)["error"]["message"])

    @mock.patch("gemini_web2api.server.generate_stream", return_value=iter(["streamed"]))
    def test_google_stream_generate_content_uses_sse(self, _generate_stream):
        status, headers, body = self.post_json(
            "/v1beta/models/gemini-3.6-flash:streamGenerateContent",
            {
                "contents": [{
                    "role": "user",
                    "parts": [{"text": "Stream this"}],
                }],
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        self.assertIn('"text": "streamed"', body)

    @mock.patch("gemini_web2api.server.generate", return_value="hello")
    def test_responses_text_stream_has_complete_event_sequence(self, _generate):
        status, headers, body = self.post_json(
            "/v1/responses",
            {
                "model": "gemini-3.6-flash",
                "input": "hello",
                "stream": True,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        events = _decode_sse(body)
        self.assertEqual(
            [event_type for event_type, _ in events],
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ],
        )
        self.assertEqual(
            [event["sequence_number"] for _, event in events],
            list(range(1, len(events) + 1)),
        )
        self.assertEqual(events[4][1]["delta"], "hello")
        self.assertEqual(events[-1][1]["response"]["status"], "completed")
        self.assertEqual(events[-1][1]["response"]["output"][0]["content"][0]["text"], "hello")

    @mock.patch("gemini_web2api.server.parse_tool_calls")
    @mock.patch("gemini_web2api.server.generate", return_value="tool output")
    def test_responses_function_call_stream_has_complete_event_sequence(
        self, _generate, parse_tool_calls
    ):
        parse_tool_calls.return_value = (
            "",
            [
                {
                    "id": "call_test",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Shanghai"}'},
                }
            ],
        )

        status, _, body = self.post_json(
            "/v1/responses",
            {
                "model": "gemini-3.6-flash",
                "input": "weather",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    }
                ],
                "stream": True,
            },
        )

        self.assertEqual(status, 200)
        events = _decode_sse(body)
        self.assertEqual(
            [event_type for event_type, _ in events],
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ],
        )
        self.assertEqual(
            [event["sequence_number"] for _, event in events],
            list(range(1, len(events) + 1)),
        )
        self.assertEqual(events[2][1]["output_index"], 0)
        self.assertEqual(events[3][1]["delta"], '{"city":"Shanghai"}')
        self.assertEqual(events[4][1]["arguments"], '{"city":"Shanghai"}')
        self.assertEqual(events[-1][1]["response"]["output"][0]["name"], "get_weather")

    @mock.patch("gemini_web2api.server.generate", return_value='```tool_call\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```')
    def test_chat_stream_tool_calls_include_index_and_argon_reasoning_effort(self, generate):
        status, _, body = self.post_json(
            "/v1/chat/completions",
            {
                "model": "gemini-4-argon",
                "reasoning_effort": "high",
                "messages": [{"role": "user", "content": "read a.py"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "description": "Read file",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "stream": True,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(generate.call_args.args[1], 8)
        self.assertEqual(generate.call_args.args[2], 0)
        ticket = json.loads(generate.call_args.args[4]["_ticket"])
        self.assertEqual(ticket[4], "95a4c0abcba24364")
        self.assertEqual(ticket[14], 8)
        self.assertEqual(ticket[15], 2)
        chunks = [
            json.loads(line[len("data: "):])
            for line in body.splitlines()
            if line.startswith("data: {")
        ]
        tool_call_delta = chunks[0]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(tool_call_delta["index"], 0)
        self.assertEqual(tool_call_delta["function"]["name"], "read_file")


class ArgonAndModelRoutingTests(unittest.TestCase):
    def setUp(self):
        self.original_config = dict(CONFIG)

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)

    def test_modular_and_single_file_models_in_sync(self):
        self.assertEqual(set(MODELS.keys()), set(single_file.MODELS.keys()))
        for name in MODELS:
            self.assertEqual(MODELS[name]["mode"], single_file.MODELS[name]["mode"])
            self.assertEqual(MODELS[name]["think"], single_file.MODELS[name]["think"])
            self.assertEqual(MODELS[name].get("variant"), single_file.MODELS[name].get("variant"))
            self.assertEqual(MODELS[name].get("ticket"), single_file.MODELS[name].get("ticket"))

    def test_argon_thinking_variants_and_reasoning_effort(self):
        name, mode, think, err, extra = resolve_model("gemini-4-argon", reasoning_effort="low")
        self.assertIsNone(err)
        self.assertEqual((name, mode, think), ("gemini-4-argon", 8, 4))
        self.assertEqual(json.loads(extra["_ticket"])[15], 1)

        name, mode, think, err, extra = resolve_model("gemini-4-argon", reasoning_effort="medium")
        self.assertIsNone(err)
        self.assertEqual((name, mode, think), ("gemini-4-argon", 8, 2))
        self.assertEqual(json.loads(extra["_ticket"])[15], 4)

        name, mode, think, err, extra = resolve_model("gemini-4-argon", reasoning_effort="high")
        self.assertIsNone(err)
        self.assertEqual((name, mode, think), ("gemini-4-argon", 8, 0))
        ticket = json.loads(extra["_ticket"])
        self.assertEqual(len(ticket), 17)
        self.assertEqual(ticket[4], "95a4c0abcba24364")
        self.assertEqual(ticket[14], 8)
        self.assertEqual(ticket[15], 2)
        headers = _build_headers(extra)
        self.assertIn("X-Goog-Ext-525001261-Jspb", headers)

        inner = _decode_payload(_build_payload("hi", mode, think, None, extra))
        self.assertEqual(inner[79], 8)
        self.assertIsNone(inner[80])
        self.assertEqual(inner[17], [[0]])

    def test_persistent_chat_window_continues_session_with_delta_only(self):
        import urllib.request
        from gemini_web2api.gemini import _chat_window, set_last_conv_ids
        CONFIG["persistent_chat"] = True
        _chat_window.reset()

        def fake_gen_turn1(prompt, model_id, think_mode, file_refs=None, extra_fields=None):
            set_last_conv_ids(("c_123", "r_456", "rc_789"))
            return '```tool_call\n{"name": "read", "arguments": {"filePath": "a.py"}}\n```'

        with mock.patch("gemini_web2api.server.generate", side_effect=fake_gen_turn1) as gen1:
            from gemini_web2api.server import GeminiHandler, ThreadedServer
            server = ThreadedServer(("127.0.0.1", 0), GeminiHandler)
            import threading
            t = threading.Thread(target=server.serve_forever, daemon=True)
            t.start()
            try:
                port = server.server_address[1]
                tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
                msgs1 = [{"role": "system", "content": "sys"}, {"role": "user", "content": "turn1"}]
                req1 = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    data=json.dumps({"model": "gemini-4-argon", "messages": msgs1, "tools": tools}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req1, timeout=5) as r1:
                    resp1 = json.loads(r1.read().decode())
                    self.assertEqual(r1.status, 200)
                self.assertNotIn(2, gen1.call_args.args[4])

                msgs2 = msgs1 + [
                    resp1["choices"][0]["message"],
                    {"role": "tool", "name": "read", "content": "file-body"},
                ]
                req2 = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    data=json.dumps({"model": "gemini-4-argon", "messages": msgs2, "tools": tools}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req2, timeout=5) as r2:
                    self.assertEqual(r2.status, 200)
                self.assertEqual(gen1.call_args.args[4][2][:3], ["c_123", "r_456", "rc_789"])
                self.assertEqual(gen1.call_args.args[0], "[Tool result for read]: file-body")
            finally:
                server.shutdown()
                server.server_close()
                _chat_window.reset()

    def test_large_tool_list_compacts_preserving_parameter_names(self):
        from gemini_web2api.tools import _compact_tool_defs
        compacted = _compact_tool_defs([
            {
                "name": "read_file",
                "description": "Line 1\nLine 2",
                "parameters": {
                    "type": "object",
                    "properties": {"filePath": {"type": "string", "description": "long"}},
                    "required": ["filePath"],
                },
            }
        ])
        self.assertEqual(compacted[0]["name"], "read_file")
        self.assertEqual(compacted[0]["description"], "Line 1")
        self.assertEqual(compacted[0]["parameters"], {"filePath": "string*"})

    def test_provider_prefixed_model_and_think_variant_sync(self):
        name, mode, think, err, extra = resolve_model("gemini-web/gemini-4-argon@think=0")
        self.assertIsNone(err)
        self.assertEqual((name, mode, think), ("gemini-4-argon", 8, 0))
        self.assertEqual(json.loads(extra["_ticket"])[15], 2)

    def test_parse_tool_calls_supports_multiple_formats_and_filters_unknown_json(self):
        clean, calls = parse_tool_calls(
            'Hello\n```json\n{"name": "bash", "args": {"command": "ls"}}\n```',
            allowed_tools={"bash"},
        )
        self.assertEqual(clean, "Hello")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "bash")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"command": "ls"})

        # Arbitrary JSON with "name"/"args" not in allowed_tools must not become a tool call
        prose = 'Example:\n```json\n{"name": "widget", "args": {"a": 1}}\n```'
        clean2, calls2 = parse_tool_calls(prose, allowed_tools={"bash"})
        self.assertEqual(calls2, [])
        self.assertEqual(clean2, prose)


if __name__ == "__main__":
    unittest.main()
