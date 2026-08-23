"""
Drishti - Configuration & Settings
==================================
Centralized configuration management for the Drishti Agentic + MCP Intelligence Layer.
Loads settings from environment variables with graceful defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root if present
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Directory paths
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Data file paths
MAIN_CSV_PATH = DATA_DIR / "Drishti_Cascade_Final_With_EMDAT.csv"
EVENT_CATALOG_PATH = DATA_DIR / "event_catalog.json"
EVENT_STORE_PATH = RESULTS_DIR / "event_store.json"
MODEL_A_OOF_PATH = RESULTS_DIR / "model_a_predictions_oof.csv"
MODEL_B_OOF_PATH = RESULTS_DIR / "model_b_predictions_oof.csv"
MODEL_C_OOF_PATH = RESULTS_DIR / "model_c_predictions_oof.csv"

# LLM settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free").strip()
DRISHTI_USE_MOCK_LLM = os.getenv("DRISHTI_USE_MOCK_LLM", "false").lower() in ("true", "1", "yes")

# GDELT settings
GDELT_BASE_URL = os.getenv("GDELT_BASE_URL", "https://api.gdeltproject.org/api/v2/doc/doc").strip()
GDELT_TIMEOUT_SECONDS = int(os.getenv("GDELT_TIMEOUT_SECONDS", "4"))
GDELT_DEFAULT_MAX_RECORDS = int(os.getenv("GDELT_DEFAULT_MAX_RECORDS", "5"))
GDELT_DEFAULT_TIMESPAN = os.getenv("GDELT_DEFAULT_TIMESPAN", "7d").strip()

# MCP Server settings
MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", "drishti-mcp-server")
MCP_SERVER_VERSION = os.getenv("MCP_SERVER_VERSION", "1.0.0")

# Common HS4 crop dictionary for quick commodity resolution
COMMODITY_TO_HS4 = {
    "rice": 1006,
    "wheat": 1001,
    "corn": 1005,
    "maize": 1005,
    "barley": 1003,
    "sorghum": 1007,
    "jowar": 1007,
    "millet": 1008,
    "bajra": 1008,
    "pulse": 713,
    "pulses": 713,
    "gram": 713,
    "chickpea": 713,
    "lentil": 713,
    "lentils": 713,
    "onion": 703,
    "onions": 703,
    "garlic": 703,
    "potato": 701,
    "potatoes": 701,
    "soybean": 1201,
    "soyabean": 1201,
    "groundnut": 1202,
    "peanut": 1202,
    "palm oil": 1511,
    "palmoil": 1511,
    "sunflower": 1206,
    "sugar": 1701,
    "sugarcane": 1701,
    "banana": 803,
    "cashew": 801,
    "cashewnut": 801,
    "tea": 902,
    "coffee": 901,
    "pepper": 904,
    "turmeric": 910,
    "ginger": 910,
    "chilli": 904,
    "chillies": 904,
}
