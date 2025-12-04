import cv2
import numpy as np
import os

def create_dummy_video(path: str, duration: int = 2, fps: int = 30):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    height, width = 240, 320
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    frames = duration * fps
    for i in range(frames):
        # Create a frame with changing color
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (i % 255, (i*2) % 255, (i*3) % 255)
        
        # Add some text
        cv2.putText(img, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(img)
        
    out.release()

if __name__ == "__main__":
    create_dummy_video("sample_data/tiny_sample.mp4")
