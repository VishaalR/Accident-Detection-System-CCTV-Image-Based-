from ultralytics import YOLO
import cv2
model = YOLO(r"D:\Vishaal R\VS Code\runs\accident_yolov8\weights\best.pt")
cap = cv2.VideoCapture(r"D:\Vishaal R\VS Code\Datasets\Realistic Highway Car Crashes #43.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("final_output.mp4", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
accident_frames = 0
threshold = 5
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, conf=0.5, verbose=False)
    accident_detected = False
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls].lower()
            if label == "accident":
                accident_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    if accident_detected:
        accident_frames += 1
    else:
        accident_frames = 0
    if accident_frames >= threshold:
        cv2.putText(frame, "ACCIDENT DETECTED", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    out.write(frame)
    cv2.imshow("Accident Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
print("Output saved as final_output.mp4")
