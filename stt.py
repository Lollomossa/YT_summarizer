# stt.py

import yt_dlp
import os
import uuid
import glob
import whisper
import pandas as pd

# CUpload Whisper model only once
model = whisper.load_model("base")

def youtube_to_text(video_url, prefix="audio"):
    print(f"🔗 Downloading audio from: {video_url}")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f"{prefix}.%(ext)s",
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    mp3_files = glob.glob(f"{prefix}*.mp3")
    if not mp3_files:
        raise FileNotFoundError("❌ Nessun file mp3 trovato dopo il download.")

    audio_file = mp3_files[0]
    print(f"🧠 Transcribing: {audio_file}")
    result = model.transcribe(audio_file)
    
    # After the transcription, delete the audio file
    os.remove(audio_file)

    return result["text"]

# --- Script to exxtract many videos and save them (optional) ---
if __name__ == "__main__":
    video_urls = [
        "https://www.youtube.com/watch?v=GmOzih6I1zs",
        "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
        "https://www.youtube.com/watch?v=36m1o-tM05g",
        "https://www.youtube.com/watch?v=5MgBikgcWnY",
        "https://www.youtube.com/watch?v=LnJwH_PZXnM",
        "https://www.youtube.com/watch?v=v1ojZKWfShQ",
        "https://www.youtube.com/watch?v=7vZmOF11P9A",
        "https://www.youtube.com/watch?v=w-HYZv6HzAs",
        "https://www.youtube.com/watch?v=aImrjNPrh30",
        "https://www.youtube.com/watch?v=eVFzbxmKNUw",
        "https://youtu.be/PFDu9oVAE-g?si=opZMItyfJOa",
        "https://www.youtube.com/watch?v=aircAruvnKk"
    ]

    labels = [
        "money",
        "music",
        "philosophy",
        "education",
        "psychology",
        "psychology",
        "public relations",
        "self-empowerment",
        "speaking",
        "self-empowerment",
        "mathematics",
        "computer science"
    ]

    data = []

    for video_url, label in zip(video_urls, labels):
        try:
            prefix = str(uuid.uuid4())
            transcript = youtube_to_text(video_url, prefix)
            data.append({
                "transcript": transcript,
                "label": label
            })
        except Exception as e:
            print(f"❌ Error on {video_url}: {e}")

    # Save the Datasets
    df = pd.DataFrame(data)
    df.to_csv("dataset.csv", index=False)
    print("✅ File 'dataset.csv' successfully created!")