"""
Unit Tests: GDELT News MCP Tool
================================
Tests response parsing, timeout handling, empty results, and error resilience.
Uses mocked network responses to guarantee fast, deterministic test execution.
"""

import unittest
from unittest.mock import patch, MagicMock
import requests

from drishti_mcp.tools.gdelt_tool import search_gdelt_news


class TestGDELTTool(unittest.TestCase):
    """Test suite for search_gdelt_news MCP tool."""

    def test_empty_query_returns_error(self):
        result = search_gdelt_news(query="")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["article_count"], 0)
        self.assertEqual(result["provenance"], "[GDELT DATA]")

    @patch("requests.get")
    def test_successful_gdelt_response_parsing(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "articles": [
                {
                    "title": "India permits wheat flour export under quota",
                    "url": "https://example.com/news/1",
                    "sourcecountry": "India",
                    "seendate": "20240315T120000Z",
                    "domain": "example.com",
                    "language": "English",
                },
                {
                    "title": "Black Sea grain deal tensions rise",
                    "url": "https://example.com/news/2",
                    "sourcecountry": "Ukraine",
                    "seendate": "20240316T083000Z",
                    "domain": "example.com",
                    "language": "English",
                }
            ]
        }
        mock_get.return_value = mock_response

        result = search_gdelt_news(query="wheat export", timespan="7d", max_records=2)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["article_count"], 2)
        self.assertEqual(len(result["articles"]), 2)
        self.assertEqual(result["articles"][0]["title"], "India permits wheat flour export under quota")
        self.assertEqual(result["articles"][0]["source_country"], "India")
        self.assertEqual(result["provenance"], "[GDELT DATA]")

    @patch("requests.get")
    def test_timeout_handled_gracefully(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")
        result = search_gdelt_news("wheat", timeout=1)
        self.assertIn(result["status"], ["timeout", "unavailable"])
        self.assertEqual(result["article_count"], 0)
        self.assertEqual(result["articles"], [])
        self.assertIn("timed out", result["error"].lower())

    @patch("requests.get")
    def test_http_error_handled_gracefully(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service Unavailable"
        mock_get.return_value = mock_response

        result = search_gdelt_news(query="palm oil tariff")

        self.assertEqual(result["status"], "unavailable")
        self.assertIn("503", result["error"])
        self.assertEqual(result["article_count"], 0)

    @patch("requests.get")
    def test_empty_article_list(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"articles": []}
        mock_get.return_value = mock_response

        result = search_gdelt_news(query="nonexistent_commodity_query_12345")
        self.assertIn(result["status"], ["no_articles", "empty"])
        self.assertEqual(result["article_count"], 0)
        self.assertEqual(result["articles"], [])


if __name__ == "__main__":
    unittest.main()
