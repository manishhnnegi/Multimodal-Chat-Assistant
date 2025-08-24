"""
config.py

Configuration module for the Multimodal Chat Assistant application.

This file centralizes all runtime settings, including:
- Device selection (CPU/GPU).
- Model identifiers (CLIP, BLIP, Gemini LLM).
- Flask server configuration (host, port, debug).
- API keys loaded from `.env`.

Usage
-----
    from config.config import Config
    print(Config.DEVICE)
    print(Config.MODELS["CLIP"])
"""

import torch
import os
from dotenv import load_dotenv

# Load .env file (from config folder)
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)


class Config:
    """
    Global configuration for the application.

    Attributes
    ----------
    DEVICE : str
        The device to use for model execution ("cuda" if GPU available, otherwise "cpu").
    MODELS : dict[str, str]
        Dictionary of model identifiers:
            - "CLIP": str, pretrained CLIP model.
            - "BLIP": str, pretrained BLIP model.
            - "GEMINI": str, Gemini LLM identifier.
    SERVER : dict[str, object]
        Flask server configuration:
            - "HOST": str, host address for the server (default "0.0.0.0").
            - "PORT": int, port number (default 5000).
            - "DEBUG": bool, debug mode toggle (default True).
    GEMINI_API_KEY : str or None
        API key for Google Gemini LLM, loaded from `.env`.
    """

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    MODELS: dict[str, str] = {
        "CLIP": "openai/clip-vit-base-patch32",
        "BLIP": "Salesforce/blip-image-captioning-base",
        "GEMINI": "gemini-2.0-flash",
    }

    SERVER: dict[str, object] = {
        "HOST": "0.0.0.0",
        "PORT": 5000,
        "DEBUG": True,
    }

    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")


if __name__ == "__main__":
    m: str = Config.MODELS["CLIP"]
    m = Config.GEMINI_API_KEY
    print(m)
