# run_pipeline.py

import argparse
import uuid
import os
import sys

from stt import youtube_to_text
from preprocessing import clean_text
from summarizer import summarize
from classifier import classify_text


def main(audio_path=None, url=None, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    if url:
        print("[0/5] Download and Transcription from YouTube...")
        prefix = str(uuid.uuid4())
        transcript = youtube_to_text(url, prefix)
    else:
        raise ValueError("Only YouTube URLs inputs accepted so far.")

    transcript_path = os.path.join(output_dir, "transcript.txt")
    with open(transcript_path, "w") as f:
        f.write(transcript)

    print("[1/5] Text preprocessing...")
    cleaned_text = clean_text(transcript)
    cleaned_path = os.path.join(output_dir, "cleaned_text.txt")
    with open(cleaned_path, "w") as f:
        f.write(cleaned_text)

    print("[2/5] Summarizing the content...")
    summary = summarize(cleaned_text)
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)

    print("[3/5] Classification of the content...")

    try:
        label = classify_text(summary)
        classification_path = os.path.join(output_dir, "classification.txt")
        with open(classification_path, "w") as f:
            f.write(label)
        print("[4/5] Classification saving completed!")
    except ValueError as e:
        print(f"❌ Classification error: {e}")
        print("Tip: make sure that classifier.pkl and vectorizer.pkl were generated together and that are coherent")

    print("[5/5] Completed! All the results were saved in:", output_dir)
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic pipeline for transcription, summary and classification of a YouTube video")
    parser.add_argument("--url", type=str, required=True, help="Video URL from which you want to extract the audio")
    parser.add_argument("--output_dir", type=str, default="output", help="outptut saving folder")
    args = parser.parse_args()

    main(url=args.url, output_dir=args.output_dir)