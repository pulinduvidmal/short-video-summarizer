
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, BartTokenizer, BartForConditionalGeneration, pipeline
from moviepy.editor import VideoFileClip
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify
import base64
import os
import io
import time

app = Flask(__name__)

# Load pre-trained models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
r_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

def extract_significant_frames(video_path, num_frames=10):
    video = VideoFileClip(video_path)
    duration = video.duration
    frame_times = np.linspace(0, duration, num_frames + 2)[1:-1]
    frames = []
    for t in frame_times:
        frame = video.get_frame(t)
        frames.append(Image.fromarray(frame))
    video.reader.close()
    video.audio.reader.close_proc()
    return frames

def generate_description(frame):
    inputs = processor(images=frame, return_tensors="pt").to(device)
    with torch.no_grad():
        out = r_model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    return description

def generate_descriptions(frames, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        descriptions = list(executor.map(generate_description, frames))
    return descriptions

def generate_summary(descriptions, max_length=130, min_length=30):
    text = " ".join(descriptions)
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    with torch.no_grad():
        summary_ids = model.generate(inputs, num_beams=4, max_length=max_length, min_length=min_length, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    video = request.files['video']
    video_path = 'uploaded_video.mp4'
    video.save(video_path)

    video_clip = VideoFileClip(video_path)
    frame_time = video_clip.duration / 3
    frame = video_clip.get_frame(frame_time)
    frame_image = Image.fromarray(frame)
    frame_image = frame_image.convert('RGB')
    frame_bytes = io.BytesIO()
    frame_image.save(frame_bytes, format='JPEG', quality=90)
    frame_base64 = base64.b64encode(frame_bytes.getvalue()).decode('utf-8')
    
    video_clip.reader.close()
    video_clip.audio.reader.close_proc()

    return jsonify({'frame': frame_base64, 'video_path': video_path})

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    video_path = data['video_path']

    frames = extract_significant_frames(video_path)
    descriptions = generate_descriptions(frames)
    summary = generate_summary(descriptions)
    
    # Adding a slight delay to ensure file operations are complete
    time.sleep(1)

    try:
        os.remove(video_path)
    except PermissionError as e:
        return jsonify({'error': f"Failed to delete {video_path}: {e}"}), 500

    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
