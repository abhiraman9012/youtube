# 1. Import libraries
import os
import re
import json
import mimetypes
import tempfile
import datetime
import base64
import subprocess
import numpy as np
import soundfile as sf
from google import genai
from google.genai import types # Need types for Content/Part/Config/SafetySetting
from IPython.display import display, Image, Audio, HTML
from PIL import Image as PILImage
from kokoro import KPipeline

# 3. --- SET API KEY IN ENVIRONMENT ---
#    Make sure this is done BEFORE running this cell.
#    e.g., os.environ['GEMINI_API_KEY'] = "YOUR_API_KEY_HERE"
# ------------------------------------

# Setting API Key
os.environ['GEMINI_API_KEY'] = "AIzaSyDNeeKDXnwGF7MYhFrnFoD9VL-ecvO5mEE"

# --- Check API Key ---
api_key_check = os.environ.get("GEMINI_API_KEY")
if not api_key_check:
    print("🛑 ERROR: Environment variable GEMINI_API_KEY is not set.")
    print("💡 TIP: Uncomment and set your API key above, or run this in a cell before running this script:")
    print("    os.environ['GEMINI_API_KEY'] = 'YOUR_API_KEY_HERE'")
    raise ValueError("API Key not found in environment.")
else:
    print(f"✅ Found API Key: ...{api_key_check[-4:]}")
#------------------------

# Define Safety Settings
safety_settings = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
]
print(f"⚙️ Defined Safety Settings: {safety_settings}")
