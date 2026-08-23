"""
Drishti - Gemini / Multi-Provider LLM Client
============================================
Thin compatibility layer delegating to the unified, multi-provider LLMRouter.
Supports text and structured JSON generation across Gemini, Groq, OpenRouter, and Offline Mock.
"""

from llm.llm_router import LLMRouter

class GeminiClient(LLMRouter):
    """
    Unified client for invoking LLMs (Gemini primary with Groq/OpenRouter fallback).
    Enforces strict non-hallucination of quantitative ML outputs.
    """
    pass
