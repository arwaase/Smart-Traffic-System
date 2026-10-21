import cv2
import os
import pandas as pd
from collections import defaultdict
from ultralytics import YOLO

def run_smart_traffic_system():
    # 1. Path to the fine-tuned model weights (available after training finishes)
    model_path = "runs/detect/train/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Warning: Fine-tuned weights not found at {model_path}. Please wait for training to finish.")
        return

    model = YOLO(model_path)

    # 2. Input test video path
    video_path = "videos/traffic_short.mp4"
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output processed video
    os.makedirs("output", exist_ok=True)
    out = cv2.VideoWriter("output/final_processed_traffic.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Crossing line coordinate
    line_y = int(height * 0.5)

    # Toll rates per vehicle type
    TOLL_RATES = {
        "car": 1.5,
        "van": 2.0,
        "truck": 2.5,
        "bus": 2.0,
        "motorcycle": 0.0,
        "traffic-police motorcycle": 0.0
    }

    # Custom colors: Red reserved for violations/police, others specific colors[cite: 1]
    CLASS_COLORS = {
        "car": (255, 191, 0),       # Blue-ish
        "van": (255, 0, 255),       # Magenta
        "truck": (0, 255, 255),     # Yellow
        "bus": (0, 255, 0),         # Green
        "motorcycle": (0, 0, 255),  # Red (Violation/Police)
        "traffic-police motorcycle": (0, 0, 255) # Red (Violation/Police)
    }
    DEFAULT_COLOR = (200, 200, 200)

    summary_stats = defaultdict(lambda: {"going": 0, "returning": 0, "total_fee": 0.0})
    previous_positions = {}
    counted_ids = set()
    total_crossed_count = 0
    report_records = []

    print("Starting Smart Traffic System Inference...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Track objects across frames
        results = model.track(frame, persist=True, tracker="botsort.yaml", conf=0.3, verbose=False)
        
        # Draw crossing line
        cv2.line(frame, (50, line_y), (width - 50, line_y), (0, 255, 255), 2)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for box, track_id, cls in zip(boxes, track_ids, classes):
                x1, y1, x2, y2 = map(int, box)
                center_y = int((y1 + y2) / 2)
                track_id = int(track_id)
                class_name = model.names[int(cls)].lower()

                # Determine direction (Going / Returning)
                direction = "ذهاب"
                if track_id in previous_positions:
                    if center_y > previous_positions[track_id]:
                        direction = "ذهاب"
                    else:
                        direction = "إياب"
                previous_positions[track_id] = center_y

                # Check if it's a violation or police motorcycle (Red color reserved)[cite: 1]
                is_violation = class_name in ["motorcycle", "traffic-police motorcycle"]
                box_color = (0, 0, 255) if is_violation else CLASS_COLORS.get(class_name, DEFAULT_COLOR)

                # Check crossing event
                if line_y - 10 < center_y < line_y + 10:
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)
                        total_crossed_count += 1
                        fee = TOLL_RATES.get(class_name, 1.5)

                        if direction == "ذهاب":
                            summary_stats[class_name]["going"] += 1
                        else:
                            summary_stats[class_name]["returning"] += 1
                        
                        summary_stats[class_name]["total_fee"] += fee

                        # Log individual record for CSV report
                        report_records.append({
                            "Vehicle_ID": track_id,
                            "Vehicle_Type": class_name,
                            "Direction": direction,
                            "Toll_Fee_USD": fee
                        })

                # Draw bounding box and label
                label = f"ID:{track_id} {class_name}" + (" [VIOLATION]" if is_violation else "")
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # Video UI Overlay: Show only crossed vehicle count (No total price on video)
        cv2.putText(frame, f"Crossed Vehicles Count: {total_crossed_count}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    # Export summary report to CSV
    df = pd.DataFrame(report_records)
    df.to_csv("traffic_summary_report.csv", index=False)
    print("Processing finished successfully! Report 'traffic_summary_report.csv' and output video generated.")

if __name__ == "__main__":
    run_smart_traffic_system()