from transformers import (
    CLIPProcessor,
    CLIPModel,
    BlipProcessor,
    BlipForConditionalGeneration,
)

from src.llms.gemini_client import Google_LLM
from config.config import Config


class ModelSetUp:
    """
    A centralized setup class for initializing and managing vision-language models
    (CLIP, BLIP) and a Large Language Model (Google Gemini) for downstream tasks
    such as image-text understanding, caption generation, and conversational AI.

    This class loads models onto the configured device (CPU/GPU) and provides
    ready-to-use processors for each model.

    Attributes
    ----------
    device : str
        The computation device (e.g., "cuda" or "cpu") defined in Config.
    clip_model : transformers.CLIPModel
        The pretrained CLIP model for image-text similarity and embeddings.
    clip_processor : transformers.CLIPProcessor
        The processor (tokenizer + feature extractor) for CLIP.
    blip_model : transformers.BlipForConditionalGeneration
        The pretrained BLIP model for image captioning and vision-language tasks.
    blip_processor : transformers.BlipProcessor
        The processor (tokenizer + feature extractor) for BLIP.
    llm : Google_LLM
        A wrapper for interacting with Google's Gemini LLM API.
    """

    def __init__(self):
        """
        Initialize the ModelSetUp instance by loading all models
        (CLIP, BLIP, and Gemini LLM) onto the configured device.
        """

        self.device = Config.DEVICE
        self._load_models()

    def _load_models(self):
        """
        Internal helper method to load all required models and processors.

        Loads:
        - CLIP model & processor for image-text understanding.
        - BLIP model & processor for image captioning/vision-language tasks.
        - Google Gemini LLM client for natural language generation.

        All models are loaded from the paths defined in Config.MODELS
        and moved to the specified computation device.
        """

        # CLIP
        self.clip_model = CLIPModel.from_pretrained(Config.MODELS["CLIP"]).to(
            self.device
        )
        self.clip_processor = CLIPProcessor.from_pretrained(Config.MODELS["CLIP"])

        # BLIP
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            Config.MODELS["BLIP"]
        ).to(self.device)
        self.blip_processor = BlipProcessor.from_pretrained(Config.MODELS["BLIP"])

        # LLM
        self.llm = Google_LLM(
            api_key=Config.GEMINI_API_KEY, model=Config.MODELS["GEMINI"]
        )
