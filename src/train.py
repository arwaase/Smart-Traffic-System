from ultralytics import YOLO

def main():
    # 1. Load the lightweight pretrained YOLO11n model
    model = YOLO("yolo11n.pt")

    # 2. Start the fine-tuning process using your custom dataset
    results = model.train(
        data="My First Project.v1i.yolov11/data.yaml"   ,     
        epochs=30,         # Number of training epochs
        imgsz=640,         # Image size for training
        batch=16,          # Batch size
        device="cpu"       # Use "0" if you have a compatible NVIDIA GPU, otherwise keep "cpu"
    )

    print("Training finished successfully and the new model weights have been saved!")

if __name__ == '__main__':
    main()