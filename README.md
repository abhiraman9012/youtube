# YouTube Story Video Generation Pipeline

This project provides an end-to-end pipeline for generating children's story videos using Gemini AI. The system generates a story prompt, creates a unique story with AI-generated images, converts the story to speech, combines everything into a video with effects, and generates SEO-friendly metadata for YouTube uploads.

## Features

- 🤖 AI-generated story prompt with customizable themes
- 🎨 High-quality image generation for each story scene
- 🔊 Text-to-speech conversion using Kokoro TTS
- 🎬 Professional video creation with dynamic effects and transitions
- 📊 SEO optimization with auto-generated titles, descriptions, and tags
- 🖼️ Custom thumbnail generation
- 📁 Google Drive integration for saving outputs (when running in Colab)

## File Structure

The codebase is organized with a clean modular structure:

- `main.py` - The main entry point that ties everything together
- `src/` - Directory containing all module files:
  - `config.py` - Configuration settings, API keys, and imports
  - `prompt_generator.py` - AI-powered story prompt generator
  - `story_utils.py` - Story text processing and cleaning utilities
  - `seo_metadata.py` - SEO metadata generation for YouTube
  - `thumbnail_generator.py` - Custom thumbnail creation
  - `video_generator.py` - Video creation with ffmpeg
  - `drive_utils.py` - Google Drive integration utilities
- `requirements.txt` - Required Python dependencies

## Installation

```bash
# Clone the repository
git clone https://github.com/abhiraman9012/youtube.git
cd youtube

# Install dependencies
pip install -r requirements.txt
```

## Usage

The Gemini API key is already provided in the `config.py` file, so you can run the code directly without needing to set your own key:

```bash
python main.py
```

If you want to use your own Gemini API key, you can edit the key in `config.py`:

```python
os.environ['GEMINI_API_KEY'] = "YOUR_API_KEY_HERE"
```

## Running in Google Colab

This project is optimized for Google Colab. To run it in Colab:

1. Create a new Colab notebook
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. Clone the repository:
   ```python
   !git clone https://github.com/abhiraman9012/youtube.git
   %cd youtube
   ```

4. Install dependencies:
   ```python
   !pip install -r requirements.txt
   ```

5. Run the main script directly (the API key is already provided in config.py):
   ```python
   %run main.py
   ```

Alternatively, you can upload the `Youtube_Pipline.ipynb` notebook to Colab, which contains all the necessary code to run the pipeline. This notebook has integrated Google Drive support for saving the generated videos.

## Pipeline Workflow

1. Generate story prompt using the thinking model (optional)
2. Generate story with images using the image generation model
3. Clean and process the story text
4. Convert story to speech using Kokoro TTS
5. Generate video with effects using ffmpeg
6. Create SEO metadata and thumbnail
7. Save outputs locally or to Google Drive

## Requirements

- Python 3.7+
- Google Gemini API key
- FFmpeg for video generation
- Internet connection for API calls
- Google account (for Colab & Google Drive integration)

## Note

A Gemini API key is already provided in the code, so you can run it immediately without any changes. If you wish to use your own API key, simply edit the key in the `config.py` file.

If running locally, make sure FFmpeg is installed on your system for video generation functionality.

The generated videos, thumbnails, and metadata are saved to Google Drive when running in Colab, or to a temporary directory when running locally.
