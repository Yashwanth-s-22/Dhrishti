"""
Unit Tests: Gemini LLM Client
=============================
Tests GeminiClient initialization, text generation, structured JSON extraction,
and deterministic mock fallback.
"""

import unittest
from llm.gemini_client import GeminiClient


class TestGeminiClient(unittest.TestCase):
    """Test suite for Gemini LLM client abstraction."""

    def setUp(self):
        self.client = GeminiClient(use_mock=True)

    def test_mock_text_generation(self):
        text = self.client.generate_text(
            prompt="Analyze the agricultural trade impact of wheat disruption.",
            system_instruction="You are the impact interpretation agent.",
        )
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 10)

    def test_mock_structured_json_event(self):
        data = self.client.generate_structured_json(
            prompt="Extract event for Russia wheat export ban.",
        )
        self.assertIsInstance(data, dict)
        self.assertEqual(data.get("country"), "RUSSIA")
        self.assertEqual(data.get("commodity"), "Wheat")
        self.assertEqual(data.get("hs4"), 1001)
        self.assertEqual(data.get("provenance"), "[LLM INFERENCE]")

    def test_mock_structured_json_mitigation(self):
        data = self.client.generate_structured_json(
            prompt="Synthesize mitigation actions for government and farmers.",
        )
        self.assertIsInstance(data, dict)
        self.assertIn("government", data)
        self.assertIn("farmers", data)
        self.assertIsInstance(data["government"], list)
        self.assertGreaterEqual(len(data["government"]), 1)

    def test_json_extraction_helper(self):
        raw_markdown = """```json
{
  "status": "success",
  "commodity": "Rice",
  "hs4": 1006
}
```"""
        parsed = GeminiClient._extract_json(raw_markdown)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["commodity"], "Rice")
        self.assertEqual(parsed["hs4"], 1006)


if __name__ == "__main__":
    unittest.main()
