"""Hugging Face Space entrypoint (Gradio SDK).

The Space runs `python app.py`; HF sets the server host/port. Inference is
ONNX Runtime on CPU and the model is pulled from the Hub at startup.
"""

from ui import demo

if __name__ == "__main__":
    # ssr_mode=False: gradio 5 SSR can glitch rendering on Spaces.
    demo.launch(ssr_mode=False)
