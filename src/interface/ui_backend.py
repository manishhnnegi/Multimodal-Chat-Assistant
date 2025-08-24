import cv2
from flask import Flask, render_template, Response, jsonify, request


class UIInterface:
    """
    Flask-based user interface backend for video capture, processing,
    summarization, and chat interactions.

    This class serves as the web interface layer, connecting frontend templates
    with the model inference pipeline (ModelInference). It provides routes for
    starting/stopping the camera, streaming video, processing frames for keyframes
    and summaries, and a chat endpoint powered by the LLM.

    Attributes
    ----------
    app : Flask
        Flask application instance serving the UI.
    model_infer : ModelInference
        Inference pipeline handling video embeddings, captions, and summarization.
    chat_memory : list[tuple[str, str]]
        Stores the last N user-assistant chat exchanges for contextual responses.
    """

    def __init__(self, model_infer):
        """
        Initialize the UI backend with a Flask app and model inference pipeline.

        Parameters
        ----------
        model_infer : ModelInference
            Inference object that provides embeddings, captioning, and text generation.
        """

        # self.app = Flask(__name__)
        self.app = Flask(
            __name__,
            template_folder="../templates",  # <-- relative path from interface/ui_backend.py
        )
        self.model_infer = model_infer
        self.chat_memory = []
        self._register_routes()

    def _register_routes(self):
        """
        Register all Flask routes for UI endpoints:
        - "/" → index page.
        - "/start" → start camera capture.
        - "/stop" → stop camera capture.
        - "/video_feed" → stream live camera feed.
        - "/process" → process frames (keyframes, captions, summaries).
        - "/chat" → handle chat interactions with LLM.
        """

        self.app.add_url_rule("/", "index", self.index)
        self.app.add_url_rule("/start", "start_capture", self.start_capture)
        self.app.add_url_rule("/stop", "stop_capture", self.stop_capture)
        self.app.add_url_rule("/video_feed", "video_feed", self.video_feed)
        self.app.add_url_rule("/process", "process", self.process)
        self.app.add_url_rule("/chat", "chat", self.chat, methods=["POST"])

    # Routes
    def index(self):
        """
        Render the main index page.

        Returns
        -------
        str
            Rendered HTML template for the UI.
        """

        return render_template("index.html")

    def start_capture(self):
        """
        Start the video capture session if not already active.

        Returns
        -------
        dict
            JSON response with capture status.
        """

        if not self.model_infer.capture_active:
            self.model_infer.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.model_infer.capture_active = True
        return {"status": "started"}

    def stop_capture(self):
        """
        Stop the video capture session and release the camera.

        Returns
        -------
        dict
            JSON response with capture status.
        """

        if self.model_infer.camera:
            self.model_infer.capture_active = False
            self.model_infer.camera.release()
            self.model_infer.camera = None
        return {"status": "stopped"}

    def video_feed(self):
        """
        Stream live video frames as an HTTP multipart response.

        Returns
        -------
        flask.Response
            Streamed video feed if active, else a plain-text error message.
        """

        if self.model_infer.capture_active:
            return Response(
                self.model_infer.generate_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )
        return Response("Camera not started yet.", mimetype="text/plain")

    def process(self):
        """
        Process a window of video frames to extract keyframes, captions,
        and a summary of events.

        Returns
        -------
        flask.Response
            JSON object containing:
            - keyframes : list[str] (base64-encoded images)
            - captions : str
            - event_log : str
            - summary : str
        """

        results = self.model_infer.process_frame_window(
            window_seconds=10, target_fps=2, n_keyframes=3
        )
        return jsonify(
            dict(zip(["keyframes", "captions", "event_log", "summary"], results))
        )

    def chat(self):
        """
        Handle a chat request by generating an assistant response using the LLM.

        The context includes:
        - Last 50 video event captions.
        - Last 5 user-assistant chat exchanges.
        - Current user query.

        Returns
        -------
        flask.Response
            JSON response with the assistant's generated reply.
        """

        user_message = request.json.get("message", "")
        context_text = ""

        if self.model_infer.event_log:
            context_text += f"Video Summary So Far: {' '.join(self.model_infer.event_log[-50:])}\n\n"
        for q, a in self.chat_memory[-5:]:
            context_text += f"User: {q}\nAssistant: {a}\n"
        context_text += f"User: {user_message}\nAssistant:"

        assistant_response = self.model_infer.generate_response(
            context_text, max_length=200
        )
        self.chat_memory.append((user_message, assistant_response))
        return jsonify({"response": assistant_response})

    def run(self, host, port, debug):
        """
        Run the Flask server for the UI interface.

        Parameters
        ----------
        host : str
            Host IP or domain (e.g., "127.0.0.1").
        port : int
            Port number for the server.
        debug : bool
            Debug mode flag.
        """

        self.app.run(host=host, port=port, debug=debug)
