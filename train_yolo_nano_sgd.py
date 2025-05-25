from ultralytics import YOLO
import multiprocessing
import torch
import os

def main():
    model_name = 'yolo11n.pt'
    data_config = 'voc_config.yaml'

    epochs = 200
    img_size = 320
    batch_size = 16
    optimizer_choice = 'SGD'
    initial_lr = 0.01
    final_lr_factor = 0.01
    momentum_val = 0.937
    weight_decay_val = 0.0005
    enable_augmentation = True
    patience_epochs = 200
    device_id = 0
    num_workers = 8
    project_name = 'runs/detect'
    run_name = 'train_yolo11n_sgd'
    save_period_val = 5
    training_seed = 42

    if isinstance(device_id, int) and not torch.cuda.is_available():
        print("CUDA specified but not available, switching to CPU.")
        device_id = 'cpu'
    elif device_id == '' and not torch.cuda.is_available():
         print("CUDA not available, training on CPU.")
         device_id = 'cpu'

    if not os.path.exists(data_config):
        print(f"ERROR: Data configuration file not found at '{data_config}'")
        return

    print(f"Loading model: {model_name}")
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return

    try:
        results = model.train(
            data=data_config,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            optimizer=optimizer_choice,
            lr0=initial_lr,
            lrf=final_lr_factor,
            momentum=momentum_val,
            weight_decay=weight_decay_val,
            augment=enable_augmentation,
            patience=patience_epochs,
            project=project_name,
            name=run_name,
            device=device_id if device_id != '' else None,
            workers=num_workers,
            save=True,
            save_period=save_period_val,
            amp=True,
            seed=training_seed,
            pretrained=True,
            exist_ok=False
        )
        print("\n--- Training Completed Successfully ---")
        print(f"Results saved to: {results.save_dir}")
        print("Note: 'last.pt' checkpoint in the results directory should contain the optimizer state.")
        print("      'best.pt' might be stripped for inference.")

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("\n--- ERROR: CUDA Out of Memory ---")
            print(f"Training failed due to insufficient GPU memory. Try reducing 'batch_size' (currently {batch_size}).")
        else:
            print(f"\n--- Training Failed with Runtime Error ---")
            print(e)
    except Exception as e:
        print(f"\n--- Training Failed with Unexpected Error ---")
        print(e)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main() 