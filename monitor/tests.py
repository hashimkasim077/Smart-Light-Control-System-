# test_camera.py
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if cap.isOpened():
    print("✅ Camera opened successfully!")
    ret, frame = cap.read()
    if ret:
        print(f"✅ Frame captured: {frame.shape}")
    else:
        print("❌ Could not read frame")
    cap.release()
else:
    print("❌ Could not open camera")