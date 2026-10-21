import cv2
import os
from ultralytics import YOLO

# مسار الفيديو القصير للتجربة
video_path = "videos/traffic_short.mp4"
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    exit()

model_path = "yolo11n.pt"
model = YOLO(model_path)

cap = cv2.VideoCapture(video_path)

# إحداثيات خط العبور (Crossing Line) لتجنب ازدواج العد
# يمكنك تعديل الاحداثيات حسب شكل الشارع في الفيديو الخاص بك
line_y = 400  # خط أفقي في منتصف الشارع تقريباً

vehicle_count = 0
counted_ids = set()

print("Starting tracking and counting test. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break

    # المعالجة المسبقة الخفيفة
    processed_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # استخدام ميزة التتبع المدمجة في Ultralytics YOLO (مثل BoT-SORT أو ByteTrack)
    results = model.track(processed_frame, persist=True, tracker="botsort.yaml", verbose=False)

    # رسم خط العبور على الإطار
    cv2.line(processed_frame, (100, line_y), (1100, line_y), (0, 255, 0), 3)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for box, track_id, cls in zip(boxes, track_ids, classes):
            x1, y1, x2, y2 = map(int, box)
            center_y = int((y1 + y2) / 2)
            
            # التحقق إذا عبرت المركبة خط العبور ولم يتم عدها مسبقاً
            if line_y - 10 < center_y < line_y + 10:
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    vehicle_count += 1

            # رسم المربع وتحديد معرف التتبع (ID)
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(processed_frame, f"ID: {int(track_id)}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # عرض العدد الإجمالي على الشاشة
    cv2.putText(processed_frame, f"Total Count: {vehicle_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Vehicle Tracking & Counting", processed_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Final Count: {vehicle_count}")