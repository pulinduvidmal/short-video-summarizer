# short-video-summarizer

This project contains a Python Flask application for summarizing videos. The application extracts significant frames from a video, generates descriptions for these frames, and then creates a summary of the video content.

## Features

- Extract significant frames from a video.
- Generate descriptions for frames using a pre-trained image captioning model.
- Generate a summary of the video using a text summarization model.
- Web interface to upload videos and view summaries.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/pulinduvidmal/short-video-summarizer.git
   cd video-summary

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   
## Usage

1. **Run the Flask application**:
   ```bash
   python video_summary.py

3. **Open your web browser and go to http://127.0.0.1:5000/ to access the application.**

4. **Upload a video file through the web interface to get a summary.**

## Files

- video_summary.py: Main Python script containing the Flask application.
- templates/index.html: HTML template for the web interface.
- requirements.txt: List of Python dependencies.

## Endpoints
- /: The main page with a form to upload a video.
- /upload: Endpoint to handle video uploads.
- /summarize: Endpoint to generate and return the video summary.


## Acknowledgments

- [Salesforce BLIP](https://huggingface.co/Salesforce/blip-image-captioning-large): Used for image captioning.
- [Facebook BART](https://huggingface.co/facebook/bart-base): Used for text summarization.
- MoviePy: Used for video processing.


