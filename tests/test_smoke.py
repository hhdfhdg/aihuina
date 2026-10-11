"""Базовые тесты: пакет импортируется, ключевые функции работают."""
import unittest

from aihuina.config import PROVIDERS, _detect_provider, resolve_effort
from aihuina.parsing import extract_tool_call, strip_reasoning


class ConfigTests(unittest.TestCase):
    def test_provider_detection(self):
        self.assertEqual(_detect_provider("https://api.openai.com/v1/chat/completions"), "openai")
        self.assertEqual(_detect_provider("http://127.0.0.1:11434/v1/chat/completions"), "ollama")
        self.assertEqual(_detect_provider("https://api.x.ai/v1/chat/completions"), "xai")
        self.assertEqual(_detect_provider("http://example.local/v1"), "custom")

    def test_all_providers_have_defaults(self):
        for key, info in PROVIDERS.items():
            self.assertTrue(info["label"], key)
            self.assertTrue(info["default_url"].startswith("http"), key)

    def test_effort_has_expected_keys(self):
        preset = resolve_effort("normal")
        for key in ("label", "max_iterations", "force_think", "preload", "note"):
            self.assertIn(key, preset)


class ParsingTests(unittest.TestCase):
    def test_plain_text_has_no_tool_call(self):
        self.assertIsNone(extract_tool_call("Привет, как дела?"))

    def test_reasoning_is_stripped(self):
        self.assertEqual(strip_reasoning("<reasoning>думаю</reasoning>ответ").strip(), "ответ")


if __name__ == "__main__":
    unittest.main()
