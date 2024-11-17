from flask import Flask, request, jsonify, send_from_directory, send_file, after_this_request
from flask_cors import CORS
import cv2
import os
import uuid
import subprocess
from process_frame import ProcessFrame  # Assuming you have this module for frame processing
from thresholds import get_thresholds_beginner  # Assuming you have this module for thresholds
from utils import get_mediapipe_pose  # Assuming you have this module for pose estimation

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# Path to save processed videos in the frontend folder
FRONTEND_VIDEOS_FOLDER = os.path.abspath(os.path.join(app.root_path, '..', 'frontend', 'public', 'videos'))
if not os.path.exists(FRONTEND_VIDEOS_FOLDER):
    os.makedirs(FRONTEND_VIDEOS_FOLDER)
    print(f"Created directory: {FRONTEND_VIDEOS_FOLDER}")

# Initialize ProcessFrame and MediaPipe Pose
thresholds = get_thresholds_beginner()
pose = get_mediapipe_pose()
process_frame = ProcessFrame(thresholds)

@app.route('/process-video', methods=['POST'])
def process_video():
    # Check if the video file is in the request
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400

    # Generate a unique ID for this video processing session
    unique_id = uuid.uuid4().hex
    temp_video_path = os.path.join(FRONTEND_VIDEOS_FOLDER, f'temp_video_{unique_id}.mp4')

    video_file.save(temp_video_path)
    print(f"Saved uploaded video to: {temp_video_path}")

    # Process video
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        return jsonify({'error': 'Error opening video file'}), 500

    # Use MP4V codec for intermediate video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Path for intermediate and final processed video
    processed_video_path = os.path.join(FRONTEND_VIDEOS_FOLDER, f'processed_intermediate_video_{unique_id}.mp4')
    final_processed_video_path = os.path.join(FRONTEND_VIDEOS_FOLDER, f'processed_video_{unique_id}.mp4')

    # Remove existing processed videos
    for video in [processed_video_path, final_processed_video_path]:
        if os.path.exists(video):
            os.remove(video)

    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))
    print(f"Processing video frames and writing to: {processed_video_path}")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            processed_frame, _ = process_frame.process(frame, pose)
            
            if processed_frame is None:
                continue

            # Ensure processed frame matches original dimensions
            if processed_frame.shape[0] != height or processed_frame.shape[1] != width:
                processed_frame = cv2.resize(processed_frame, (width, height))

            out.write(processed_frame)
    finally:
        cap.release()
        out.release()

    # Delete the uploaded temporary video after processing
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
        print(f"Deleted temporary video: {temp_video_path}")

    # Convert processed video to H.264 format using FFmpeg
    try:
        ffmpeg_command = f'ffmpeg -i "{processed_video_path}" -vcodec libx264 -acodec aac -strict -2 "{final_processed_video_path}"'
        subprocess.run(ffmpeg_command, shell=True, check=True)
        print(f"Converted video saved to: {final_processed_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        return jsonify({'error': 'Video conversion failed'}), 500

    # Delete intermediate processed video after conversion
    if os.path.exists(processed_video_path):
        os.remove(processed_video_path)
        print(f"Deleted intermediate video: {processed_video_path}")

    # Check if the final processed video exists
    if not os.path.exists(final_processed_video_path):
        return jsonify({'error': 'Final processed video was not created'}), 500

    # Return the URL of the processed video for download
    processed_video_url = f'https://fitness1-mfoj.onrender.com/download-video/{os.path.basename(final_processed_video_path)}'
    return jsonify({'message': 'Video processed and converted successfully', 'videoUrl': processed_video_url})

@app.route('/download-video/<path:filename>', methods=['GET'])
def download_video(filename):
    video_path = os.path.join(FRONTEND_VIDEOS_FOLDER, filename)

    # Ensure file deletion after request completes
    @after_this_request
    def remove_file(response):
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"Deleted processed video after download: {video_path}")
            except Exception as e:
                print(f"Error deleting file {video_path}: {e}")
        return response

    # Send the video file as an attachment for download
    return send_file(video_path, as_attachment=True)

@app.route('/videos/<path:filename>', methods=['GET'])
def serve_video(filename):
    print(f"Serving video from directory: {FRONTEND_VIDEOS_FOLDER}")
    return send_from_directory(FRONTEND_VIDEOS_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
