import os
from google import genai
from google.genai import types
from dotenv import load_dotenv, dotenv_values


class Google_LLM:
    """
    Wrapper class for interacting with Google's Gemini LLM API.

    Provides utility methods for:
    - Single-prompt text generation.
    - Initializing and maintaining a persistent chat session.
    - Exchanging messages in a conversational context.
    - Retrieving chat history.

    Attributes
    ----------
    client : genai.Client
        Google Generative AI client for sending requests to the Gemini API.
    model : str
        The name of the Gemini model to use (default: "gemini-2.0-flash").
    chat : genai.Chat or None
        Active chat session object, initialized with `start_chat()`.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        """
        Initialize the Google LLM client with API key and model.

        Parameters
        ----------
        api_key : str
            API key for authenticating with Google Gemini API.
        model : str, optional
            Model name to use for generation (default: "gemini-2.0-flash").
        """
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.chat = None

    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.3,
        top_p: float = 0.95,
        max_tokens: int = 100,  # 512,
    ) -> str:
        """
        Generate a single text response for a given prompt.

        Parameters
        ----------
        prompt : str
            Input text prompt for the model.
        temperature : float, optional
            Sampling temperature for controlling randomness (default: 0.3).
        top_p : float, optional
            Nucleus sampling parameter for probability mass (default: 0.95).
        max_tokens : int, optional
            Maximum number of tokens in the response (default: 100).

        Returns
        -------
        str
            Generated text response.
        """

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text

    def start_chat(self):
        """
        Initialize a persistent chat session with the LLM.

        Notes
        -----
        Must be called before using `chat_message()` or `get_history()`.
        """

        self.chat = self.client.chats.create(model=self.model)

    def chat_message(self, message: str) -> str:
        """
        Send a message within an ongoing chat session.

        Parameters
        ----------
        message : str
            The user message to send.

        Returns
        -------
        str
            The assistant's reply.

        Raises
        ------
        ValueError
            If no chat session is active. Call `start_chat()` first.
        """
        if not self.chat:
            raise ValueError("Chat session not started. Call start_chat() first.")
        response = self.chat.send_message(message=message)
        return response.text

    def get_history(self):
        """
        Retrieve the history of the current chat session.

        Returns
        -------
        list[tuple[str, str]]
            List of (role, text) pairs representing the conversation turns.

        Raises
        ------
        ValueError
            If no chat session is active.
        """
        if not self.chat:
            raise ValueError("No chat session active.")
        return [(turn.role, turn.parts[0].text) for turn in self.chat.get_history()]


if __name__ == "__main__":
    load_dotenv()

    # accessing and printing value

    api_key = os.getenv("GEMINI_API_KEY")

    llm = Google_LLM(api_key)

    # Single text generation
    story = llm.generate_text("what comes after monday?")
    print("Story:\n", story)
