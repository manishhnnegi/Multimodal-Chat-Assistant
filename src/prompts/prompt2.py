SUMMARY_PROMPT_TEMPLATE = """You are given the following reference captions extracted from key frames of a video:
{caption_response}

Your task:
- Analyze the captions and identify specific events in the video (e.g., vehicle movements, pedestrian actions, traffic signals, or other context-specific actions).
- Check these events against pre-defined guidelines or rules (for example, traffic rules if it's a traffic video).
- Detect and report any guideline adherence or violations.
- Summarize the main events, key actions, and context of the video.
- Highlight important violations with approximate timestamps if available.
- Provide a clear, concise and breif summary in natural language.

Format your output as:

Final Answer: <your concise summary highlighting events and guideline adherence/violations here>
"""
