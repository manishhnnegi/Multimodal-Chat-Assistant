import os
import time
import base64
import cv2
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from src.prompts.prompt2 import SUMMARY_PROMPT_TEMPLATE


class ModelInference:
    """
    A pipeline class for performing video-based vision-language inference using CLIP,
    BLIP, and a Large Language Model (LLM).

    This class provides utilities for:
    - Extracting embeddings from video frames using CLIP.
    - Selecting keyframes via clustering.
    - Generating captions for frames using BLIP.
    - Summarizing events using an LLM (Google Gemini).
    - Processing live video streams and maintaining an event log.

    Attributes
    ----------
    clip_model : transformers.CLIPModel
        Pretrained CLIP model for image embeddings.
    clip_processor : transformers.CLIPProcessor
        Processor (tokenizer + feature extractor) for CLIP.
    blip_model : transformers.BlipForConditionalGeneration
        Pretrained BLIP model for image captioning.
    blip_processor : transformers.BlipProcessor
        Processor (tokenizer + feature extractor) for BLIP.
    device : str
        Computation device ("cuda" or "cpu").
    llm : Google_LLM
        Large Language Model client for text summarization/response generation.
    event_log : list[str]
        A running log of generated captions/events.
    capture_active : bool
        Indicates whether video capture is active.
    camera : cv2.VideoCapture
        OpenCV camera capture object.
    """

    def __init__(self, setup):
        """
        Initialize the inference pipeline with preloaded models.

        Parameters
        ----------
        setup : ModelSetUp
            An instance of the ModelSetUp class containing initialized
            CLIP, BLIP, and LLM objects along with the device configuration.
        """

        self.clip_model = setup.clip_model
        self.clip_processor = setup.clip_processor
        self.blip_model = setup.blip_model
        self.blip_processor = setup.blip_processor
        self.device = setup.device
        self.llm = setup.llm

        self.event_log = []
        self.capture_active = False
        self.camera = None

    # -----------------------------
    # Embeddings & Captioning
    # -----------------------------
    def get_clip_embeddings(self, frames, batch_size=16):
        """
        Compute CLIP image embeddings for a batch of frames.

        Parameters
        ----------
        frames : list[np.ndarray]
            List of image frames in RGB format.
        batch_size : int, optional
            Batch size for processing frames (default: 16).

        Returns
        -------
        np.ndarray
            Normalized image embeddings of shape (N, D),
            where D is the embedding dimension.
        """

        all_embs = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            inputs = self.clip_processor(
                images=batch, return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                feats = self.clip_model.get_image_features(**inputs)
                feats = torch.nn.functional.normalize(feats, dim=-1)
            all_embs.append(feats.cpu().numpy())
        return np.vstack(all_embs) if all_embs else np.zeros((0, 512), dtype=np.float32)

    def select_keyframes(self, embeddings, frames, n_clusters=3):
        """
        Select representative keyframes using KMeans clustering.

        Parameters
        ----------
        embeddings : np.ndarray
            Image embeddings corresponding to frames.
        frames : list[np.ndarray]
            List of image frames.
        n_clusters : int, optional
            Number of clusters/keyframes to extract (default: 3).

        Returns
        -------
        list[np.ndarray]
            A subset of frames chosen as keyframes.
        """

        if len(frames) <= n_clusters:
            return frames
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(embeddings)
        keyframes = []
        for c in range(n_clusters):
            cluster_indices = np.where(kmeans.labels_ == c)[0]
            distances = np.linalg.norm(
                embeddings[cluster_indices] - kmeans.cluster_centers_[c], axis=1
            )
            keyframes.append(frames[cluster_indices[np.argmin(distances)]])
        return keyframes

    def caption_frames(self, frames, max_len=30):
        """
        Generate captions for frames using the BLIP model.

        Parameters
        ----------
        frames : list[np.ndarray]
            List of image frames (RGB).
        max_len : int, optional
            Maximum caption length (default: 30).

        Returns
        -------
        list[str]
            Generated captions for the frames.
        """

        captions = []
        for frame in frames:
            inputs = self.blip_processor(images=frame, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=max_len)
            cap = self.blip_processor.decode(out[0], skip_special_tokens=True)
            captions.append(cap)
        return captions

    def summarize_event_log(self, max_events=60):
        """
        Summarize recent events from the event log using an LLM.

        Parameters
        ----------
        max_events : int, optional
            Number of recent events to include in the summary (default: 60).

        Returns
        -------
        str
            Generated summary text.
        """

        if not self.event_log:
            return "No events yet."
        captions_text = "\n ".join(self.event_log[-max_events:])
        final_prompt = SUMMARY_PROMPT_TEMPLATE.format(caption_response=captions_text)
        return self.llm.generate_text(final_prompt)

    def generate_response(self, prompt, max_length=200):
        """
        Generate a text response from the LLM for a given prompt.

        Parameters
        ----------
        prompt : str
            Input text prompt.
        max_length : int, optional
            Maximum length of the generated response (default: 200).

        Returns
        -------
        str
            Generated response text.
        """

        return self.llm.generate_text(prompt)

    # -----------------------------
    # Video Processing
    # -----------------------------
    def generate_frames(self):
        """
        Generator that yields JPEG-encoded frames from an active camera feed.

        Yields
        ------
        bytes
            Encoded video frame in HTTP multipart format.
        """

        while self.capture_active and self.camera:
            success, frame = self.camera.read()
            if not success:
                continue
            _, buffer = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    def process_frame_window(self, window_seconds=5, target_fps=2, n_keyframes=3):
        """
        Process a sliding window of video frames to extract keyframes,
        generate captions, and summarize events.

        Parameters
        ----------
        window_seconds : int, optional
            Duration of the frame window to capture (default: 5 seconds).
        target_fps : int, optional
            Target frame rate for frame sampling (default: 2).
        n_keyframes : int, optional
            Number of keyframes to select (default: 3).

        Returns
        -------
        tuple
            - keyframes_b64 : list[str]
                Base64-encoded JPEG keyframes.
            - captions : str
                Concatenated captions for the keyframes.
            - recent_events : str
                Last 50 events from the event log.
            - summary : str
                LLM-generated summary of recent events.
        """

        if not self.camera or not self.capture_active:
            return [], "No frames", "\n".join(self.event_log[-50:]), "No events yet"

        frames, interval = [], 1.0 / target_fps
        last_capture_time = start_time = time.time()

        while time.time() - start_time < window_seconds:
            success, frame = self.camera.read()
            if not success:
                continue
            now = time.time()
            if now - last_capture_time >= interval:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                last_capture_time = now

        if not frames:
            return [], "No frames", "\n".join(self.event_log[-50:]), "No events yet"

        emb = self.get_clip_embeddings(frames)
        keyframes = self.select_keyframes(emb, frames, n_clusters=n_keyframes)
        captions = self.caption_frames(keyframes)
        self.event_log.extend(captions)
        summary = self.summarize_event_log(max_events=50)

        keyframes_b64 = []
        for f in keyframes:
            _, buffer = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            keyframes_b64.append(base64.b64encode(buffer).decode("utf-8"))

        return (
            keyframes_b64,
            "; ".join(captions),
            "\n".join(self.event_log[-50:]),
            summary,
        )
