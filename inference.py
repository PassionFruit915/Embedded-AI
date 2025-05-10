import os
import sys
import numpy as np
import cv2
import torch
import tiny_cnn

# Model path configuration
FULL_MODEL_PATH = 'results/improved_gesture_recognition_raspberry_pi/improved_gesture_recognition_full.pt'
QUANT_MODEL_PATH = 'results/improved_gesture_recognition_raspberry_pi/improved_gesture_recognition_quantized.pt'

# Prefer quantized model if available
if os.path.exists(QUANT_MODEL_PATH):
    print(f"Quantized model detected, using: {QUANT_MODEL_PATH}")
    MODEL_PATH = QUANT_MODEL_PATH
else:
    MODEL_PATH = FULL_MODEL_PATH

NUM_CHANNELS = 19
HEIGHT = 256
WIDTH = 256
VIDEO_EXTS = ['.mp4', '.avi', '.mov', '.mkv']

# Register model class to prevent deserialization failure
sys.modules['__main__'].ImprovedGrayMultiChannelTinyVGG = tiny_cnn.ImprovedGrayMultiChannelTinyVGG
sys.modules['__main__'].ConvReLU = tiny_cnn.ConvReLU


def load_model(path: str):
    model = torch.load(path, map_location='cpu', weights_only=False)
    model.eval()
    return model


def extract_frames(video_path: str, num_frames: int):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total-1, num_frames, dtype=int) if total > num_frames else list(range(total))
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        frames.append(gray.astype(np.float32) / 255.0)
    cap.release()
    # Pad frames if needed
    while len(frames) < num_frames and frames:
        frames.append(frames[-1])
    return np.stack(frames, axis=0)


def preprocess(frames: np.ndarray):
    tensor = torch.from_numpy(frames).unsqueeze(0)  # (1, C, H, W)
    return tensor


def infer(model, input_tensor: torch.Tensor):
    out = model(input_tensor)
    probs = torch.softmax(out, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    conf = probs[0, pred].item()
    return pred, conf


def process_video_file(model, video_path: str):
    frames = extract_frames(video_path, NUM_CHANNELS)
    if frames.size == 0:
        return f"Failed to extract frames from {os.path.basename(video_path)}"
    inp = preprocess(frames)
    pred, conf = infer(model, inp)
    return f"{os.path.basename(video_path)} -> Gesture_{pred+1} (Confidence: {conf:.4f})"


def find_video_files(root_path: str):
    videos = []
    for dirpath, _, filenames in os.walk(root_path):
        for fn in sorted(filenames):
            if os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
                videos.append(os.path.join(dirpath, fn))
    return videos


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(__file__)} <video file or directory>")
        sys.exit(1)

    path = sys.argv[1]
    if os.path.isfile(path) and os.path.splitext(path)[1].lower() in VIDEO_EXTS:
        video_list = [path]
    elif os.path.isdir(path):
        video_list = find_video_files(path)
    else:
        print(f"Invalid path or unsupported file format: {path}")
        sys.exit(1)

    model = load_model(MODEL_PATH)
    print(f"Starting batch inference on {len(video_list)} video(s) using model: {MODEL_PATH}")
    for v in video_list:
        print(process_video_file(model, v))


if __name__ == '__main__':
    main()
