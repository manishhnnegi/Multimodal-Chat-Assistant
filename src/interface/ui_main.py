from config.config import Config
from src.models.model_setup import ModelSetUp
from src.models.model_infer import ModelInference

# from models.model_setup import ModelSetUp
# from models.model_infer import ModelInference
from src.interface.ui_backend import UIInterface


class UIStarter:
    """
    Entry point for initializing and launching the application UI.

    This class sets up:
    - Global configuration (Config).
    - Vision-language models (via ModelSetUp).
    - Inference pipeline (via ModelInference).
    - User interface backend (via UIInterface).

    It provides methods to launch the UI server with host, port,
    and debug options from the configuration.

    Attributes
    ----------
    config : Config
        Application configuration object containing server and model settings.
    model_setup : ModelSetUp
        Object responsible for loading CLIP, BLIP, and LLM models.
    model_infer : ModelInference
        Inference pipeline that connects vision-language models with processing logic.
    ui_interface : UIInterface
        Backend interface for serving the UI and handling user interactions.
    """

    def __init__(self):
        """
        Initialize the UIStarter instance by setting up configuration,
        models, inference pipeline, and UI backend.
        """

        self.config = Config()
        self.model_setup = ModelSetUp()
        self.model_infer = ModelInference(self.model_setup)
        self.ui_interface = UIInterface(self.model_infer)

    def launch(self):
        """
        Launch the UI server with the configured host, port, and debug options.

        This starts the backend interface for user interaction.
        """

        self.ui_interface.run(
            host=self.config.SERVER["HOST"],
            port=self.config.SERVER["PORT"],
            debug=self.config.SERVER["DEBUG"],
        )

    @classmethod
    def start(cls):
        """
        Convenience method to create an instance of UIStarter
        and immediately launch the UI server.
        """

        run = cls()
        run.launch()
