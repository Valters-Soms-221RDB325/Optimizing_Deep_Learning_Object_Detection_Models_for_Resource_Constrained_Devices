from ultralytics import YOLO
import multiprocessing

def main():
    model = YOLO('yolo11n.pt')

    results = model.train(
        data='VOC2007.yaml',
        epochs=100,
        imgsz=320,
        optimizer='Adamax',
        lr0=0.01,
        lrf=0.01,
        weight_decay=0.0005,
        batch=32,
        device=0,
        save=True,
        save_period=1,
        verbose=True
    )


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()