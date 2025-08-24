"""
app.py

Main entry point for launching the Flask-based  Multimodal Chat Assistant application.

This script initializes the UI layer (`UIStarter`), which in turn sets up:
- Model configuration and loading (CLIP, BLIP, Gemini LLM).
- Video inference pipeline for keyframe extraction, captioning, summarization and Chat Assistant.
- Flask web server for UI interaction (live video, summaries, chat).

Usage
-----
Run the application with:

    python app.py

The server configuration (host, port, debug mode) is defined in `config/config.py`.
"""

from src.interface.ui_main import UIStarter


def main():
    """
    Entry point function to start the Flask application.

    Calls `UIStarter.start()`, which initializes models,
    sets up inference, and launches the UI interface.
    """

    UIStarter.start()


if __name__ == "__main__":
    main()
