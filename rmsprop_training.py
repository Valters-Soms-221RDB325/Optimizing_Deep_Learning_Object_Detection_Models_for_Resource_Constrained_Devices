from ultralytics import YOLO
import multiprocessing

def main():
    model = YOLO("yolo11n.pt")

    results = model.train(
        data="VOC2007.yaml",
        epochs=100,
        imgsz=320,
        batch=32,
        save=True,
        save_period=1,
        optimizer="RMSProp",
        lr0=0.001,
        weight_decay=0.0005
    )

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main() 