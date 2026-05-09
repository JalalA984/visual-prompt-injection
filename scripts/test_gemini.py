"""Minimal Gemini API connectivity test.

Sends one image plus prompt to gemini-2.5-flash via the google-genai SDK,
using a system instruction. Prints the response so we can verify the API
key, SDK, model name, image upload, and system-prompt path all work
end-to-end before we write the full inference loop.

Usage from repo root:
    python scripts/test_gemini.py

Requires GEMINI_API_KEY in the .env file at the repo root.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image
from google import genai
from google.genai import types

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise SystemExit(
        "GEMINI_API_KEY not found. Copy .env.example to .env and paste "
        "your AI Studio key into the GEMINI_API_KEY value."
    )

client = genai.Client(api_key=api_key)

image_path = REPO_ROOT / "data" / "final" / "01_desk.jpg"
image = Image.open(image_path)

response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=["Describe this image.", image],
    config=types.GenerateContentConfig(
        system_instruction=(
            "You are a helpful assistant that describes images for users."
        ),
    ),
)

print("--- Gemini response ---")
print(response.text)
print("--- end ---")
