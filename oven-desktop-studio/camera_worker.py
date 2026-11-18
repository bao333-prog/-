"""Send compressed frames from a Windows camera to the desktop preview."""
import struct
import sys
import time

import cv2


def stream(index):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        camera = cv2.VideoCapture(index, backend)
        try:
            if not camera.isOpened():
                continue
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            started = False
            misses = 0
            while misses < 20:
                ok, frame = camera.read()
                if not ok:
                    misses += 1
                    time.sleep(0.05)
                    continue
                misses = 0
                encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not encoded:
                    raise RuntimeError("Could not encode camera frame")
                data = jpeg.tobytes()
                output = sys.stdout.buffer
                output.write(struct.pack("<I", len(data)))
                output.write(data)
                output.flush()
                started = True
                time.sleep(0.04)
            if started:
                return 3
        finally:
            camera.release()
    return 2


if __name__ == "__main__":
    sys.exit(stream(int(sys.argv[1])))
