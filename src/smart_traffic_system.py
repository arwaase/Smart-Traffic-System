
import cv2
import csv
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO


# =========================================================
# 1. Project paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# إذا كان هذا الملف داخل src، نرجع إلى المجلد الرئيسي للمشروع.
if BASE_DIR.name.lower() == "src":
    PROJECT_DIR = BASE_DIR.parent
else:
    PROJECT_DIR = BASE_DIR

MODEL_PATH = (
    PROJECT_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
)

VIDEO_PATH = PROJECT_DIR / "videos" / "traffic_short.mp4"

OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "final_processed_traffic.mp4"
OUTPUT_CSV = OUTPUT_DIR / "final_traffic_report.csv"


# =========================================================
# 2. Vehicle classes and toll rates
# =========================================================

TOLL_RATES = {
    "car": 1.0,
    "van": 1.5,
    "truck": 2.5,
    "bus": 2.0,
    "motorcycle": 0.0,
    "traffic-police motorcycle": 0.0,
}

# ألوان OpenCV بترتيب BGR
COLORS = {
    "car": (255, 0, 0),
    "van": (0, 255, 255),
    "truck": (255, 128, 0),
    "bus": (255, 0, 255),
    "motorcycle": (0, 0, 255),
    "traffic-police motorcycle": (255, 255, 255),
}

# أسماء الفئات المتوقعة التي قد تظهر في النموذج.
# يجب مطابقة هذه الأسماء مع model.names الفعلية.
CLASS_ALIASES = {
    "car": "car",
    "van": "van",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "traffic-police motorcycle": "traffic-police motorcycle",
    "police motorcycle": "traffic-police motorcycle",
}


def normalize_class_name(name):
    """توحيد اسم الفئة دون افتراض أن كل دراجة تابعة للشرطة."""
    name = str(name).strip().lower().replace("_", " ")
    name = " ".join(name.split())
    return CLASS_ALIASES.get(name, name)


def is_police_motorcycle(class_name):
    return class_name == "traffic-police motorcycle"


def run_smart_traffic_system():

    # =====================================================
    # 3. Verify and load the trained model
    # =====================================================

    print("Model path:", MODEL_PATH)
    print("Model exists:", MODEL_PATH.exists())

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nTrained model was not found:\n{MODEL_PATH}\n"
            "Check the location of best.pt before running again."
        )

    model = YOLO(str(MODEL_PATH))

    print("\nLoaded model classes:")
    for class_id, class_name in model.names.items():
        print(f"  {class_id}: {class_name}")

    print("\nIMPORTANT: Check that the model contains the vehicle classes "
          "you intend to detect.")

    # =====================================================
    # 4. Open video
    # =====================================================

    print("\nVideo path:", VIDEO_PATH)

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"\nVideo was not found:\n{VIDEO_PATH}\n"
            "Check the video filename and its folder."
        )

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Could not open the input video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create the output video.")

    # =====================================================
    # 5. Counting and tracking settings
    # =====================================================

    # غيّري هذه النسبة لاحقاً إذا كان خط العبور في مكان مختلف.
    line_y = int(height * 0.60)

    # الاحتفاظ بتاريخ آخر موضع لكل مركبة.
    previous_positions = {}

    # منع عدّ المركبة نفسها أكثر من مرة.
    counted_ids = set()

    stats = {
        cls: {"count": 0, "revenue": 0.0}
        for cls in TOLL_RATES
    }

    violation_ids = set()
    frame_number = 0

    print("\nStarting video processing...")

    # =====================================================
    # 6. Process frames
    # =====================================================

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1
            annotated = frame.copy()

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.15,
                imgsz=640,
                verbose=False,
            )

            # رسم خط العبور.
            cv2.line(
                annotated,
                (0, line_y),
                (width, line_y),
                (0, 255, 255),
                2,
            )

            cv2.putText(
                annotated,
                "Crossing Line",
                (20, max(25, line_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            boxes_result = results[0].boxes

            if boxes_result is not None and len(boxes_result) > 0:

                boxes = boxes_result.xyxy.cpu().numpy()
                classes = boxes_result.cls.cpu().numpy()

                if boxes_result.id is not None:
                    track_ids = boxes_result.id.cpu().numpy()
                else:
                    track_ids = None

                for index, (box, cls) in enumerate(zip(boxes, classes)):

                    x1, y1, x2, y2 = map(int, box)

                    class_id = int(cls)
                    raw_name = model.names[class_id]
                    class_name = normalize_class_name(raw_name)

                    # إذا لم يتوفر ID فلا يمكن ضمان العد مرة واحدة.
                    track_id = (
                        int(track_ids[index])
                        if track_ids is not None
                        else None
                    )

                    # اختيار لون الفئة.
                    color = COLORS.get(class_name, (0, 255, 0))

                    # الدراجة العادية مخالفة وفق قواعد المشروع.
                    # لا نعتبر الدراجة مخالفة إلا إذا كانت الفئة
                    # المتوقعة للدراجة موجودة بالفعل في النموذج.
                    if (
                        class_name == "motorcycle"
                        and not is_police_motorcycle(class_name)
                    ):
                        color = (0, 0, 255)

                        if track_id is not None:
                            violation_ids.add(track_id)

                    # رسم المربع والاسم.
                    label = class_name

                    if track_id is not None:
                        label = f"ID {track_id}: {class_name}"

                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2,
                    )

                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

                    # لا نعدّ الفئات غير المسجلة في جدول الرسوم.
                    if class_name not in stats:
                        continue

                    # نحتاج ID موثوقاً لمنع تكرار العد.
                    if track_id is None:
                        continue

                    # مركز المركبة عمودياً.
                    center_y = int((y1 + y2) / 2)

                    previous_y = previous_positions.get(track_id)

                    # احتساب العبور عند انتقال المركز من أحد جانبي
                    # الخط إلى الجانب الآخر.
                    crossed_line = (
                        previous_y is not None
                        and (
                            (previous_y < line_y <= center_y)
                            or
                            (previous_y > line_y >= center_y)
                        )
                    )

                    if (
                        crossed_line
                        and track_id not in counted_ids
                    ):
                        counted_ids.add(track_id)

                        stats[class_name]["count"] += 1
                        stats[class_name]["revenue"] += (
                            TOLL_RATES[class_name]
                        )

                    previous_positions[track_id] = center_y

            # =================================================
            # 7. Dashboard
            # =================================================

            total_revenue = sum(
                item["revenue"] for item in stats.values()
            )

            total_count = sum(
                item["count"] for item in stats.values()
            )

            overlay = annotated.copy()

            cv2.rectangle(
                overlay,
                (10, 10),
                (410, 125),
                (0, 0, 0),
                -1,
            )

            annotated = cv2.addWeighted(
                overlay, 0.60, annotated, 0.40, 0
            )

            cv2.putText(
                annotated,
                "Ein Al-Masar Traffic Dashboard",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated,
                f"Vehicles counted: {total_count}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated,
                f"Revenue: ${total_revenue:.2f}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated,
                f"Violation IDs: {len(violation_ids)}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
            )

            writer.write(annotated)

            # طباعة التقدم كل 100 إطار.
            if frame_number % 100 == 0:
                print(
                    f"Processed {frame_number} frames; "
                    f"counted {total_count} vehicles."
                )

    finally:
        cap.release()
        writer.release()

    # =====================================================
    # 8. Export CSV report
    # =====================================================

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        fieldnames = [
            "Vehicle Type",
            "Total Count",
            "Toll Rate",
            "Total Revenue",
        ]

        report = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        report.writeheader()

        for class_name, data in stats.items():
            report.writerow({
                "Vehicle Type": class_name,
                "Total Count": data["count"],
                "Toll Rate": TOLL_RATES[class_name],
                "Total Revenue": round(data["revenue"], 2),
            })

    print("\nProcessing completed.")
    print("Output video:", OUTPUT_VIDEO)
    print("CSV report:", OUTPUT_CSV)
    print("Total counted vehicles:", sum(
        item["count"] for item in stats.values()
    ))
    print("Total revenue: $", round(sum(
        item["revenue"] for item in stats.values()
    ), 2))


if __name__ == "__main__":
    run_smart_traffic_system()
