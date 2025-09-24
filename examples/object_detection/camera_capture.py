import cv2
import os
import time
from pyorbbecsdk import *
from my_utils import frame_to_bgr_image  # utility ของ Orbbec SDK แปลง frame → numpy BGR

SAVE_DIR = "color_capture"
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    pipeline = Pipeline()

    # เปิดเฉพาะ COLOR stream
    config = Config()
    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = color_profiles.get_default_video_stream_profile()
    config.enable_stream(color_profile)

    pipeline.start(config)
    print("Camera started. Press 's' to save image, 'q' to quit.")

    while True:
        frames = pipeline.wait_for_frames(100)
        if not frames:
            continue

        frames = frames.as_frame_set()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # แปลง frame เป็น numpy BGR
        color_bgr = frame_to_bgr_image(color_frame)
        if color_bgr is None or color_bgr.size == 0:
            continue  # ป้องกัน error imshow

        cv2.imshow("Color", color_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):  # save image
            ts = int(time.time())
            img_path = os.path.join(SAVE_DIR, f"color_{ts}.jpg")
            cv2.imwrite(img_path, color_bgr)
            print(f"Saved image: {img_path}")

        elif key in (27, ord("q")):  # ESC หรือ q เพื่อออก
            break

    cv2.destroyAllWindows()
    pipeline.stop()

if __name__ == "__main__":
    main()
