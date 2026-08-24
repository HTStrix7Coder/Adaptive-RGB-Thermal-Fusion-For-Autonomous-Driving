import sys
import os

# Insert custom multi-modal ultralytics source at the beginning of sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ultralytics_source')))

# pyrefly: ignore [missing-import]
from ultralytics import YOLO

def main():
    print("🚀 Starting Dual-Stream RGB+Thermal YOLO26 Training")
    print(f"Using ultralytics from: {sys.modules['ultralytics'].__file__}")
    
    # ---------------------------------------------------------
    # MLflow Tracking Integration
    # Docs: https://docs.ultralytics.com/integrations/mlflow/
    # ---------------------------------------------------------
    from ultralytics import settings
    import mlflow
    
    # Ensure MLflow is enabled in Ultralytics settings
    settings.update({"mlflow": True})
    
    # Explicitly set the experiment name
    mlflow.set_experiment("Dual_YOLO26_Training")

    checkpoint_path = 'runs/detect/checkpoints/DualYOLO26s_6ch_v4_SEContext/weights/last.pt'
    
    # Callback to allow graceful pausing
    def check_pause_file(trainer):
        if os.path.exists('pause.txt'):
            print("\n⏸️ Pause signal detected ('pause.txt'). Stopping training gracefully at the end of this epoch...")
            trainer.stop = True
            os.remove('pause.txt')
            print("Checkpoint saved. You can resume later by running this script again.\n")

    if os.path.exists(checkpoint_path):
        print(f"\n🔄 Found existing checkpoint: {checkpoint_path}")
        print("Resuming training from the last saved epoch...")
        model = YOLO(checkpoint_path)
        
        # Clean up stale non-leaf tensors from the checkpoint
        # (prevents deepcopy crash in ModelEMA)
        det = model.model
        for mod in det.modules():
            if hasattr(mod, 'last_attention_weights'):
                delattr(mod, 'last_attention_weights')
        if hasattr(det, '_attn_cache'):
            det._attn_cache = []
        print("  ✓ Cleaned up stale checkpoint tensors")
        
        model.add_callback("on_train_epoch_end", check_pause_file)
        
        # Calling train with resume=True automatically pulls all previous hyperparameters 
        # (epochs, batch size, learning rate schedules) from the checkpoint file!
        results = model.train(resume=True)
    else:
        print("\nStarting fresh training run...")
        # 1. Build model from scratch with 6 channels
        model = YOLO('Config/yolo26s_6ch.yaml')
        model.add_callback("on_train_epoch_end", check_pause_file)
        
        # 2. Train with explicit pretrained weights so Ultralytics doesn't download the nano default
        results = model.train(
            pretrained='Models/yolo26s.pt',
            data='Config/dataset_dual.yaml',
            epochs=100,
            patience=50,
            imgsz=640,
            batch=10,
            workers=8,
            device=0,
            project='checkpoints',
            name='DualYOLO26s_6ch_v4_SEContext',
            optimizer='AdamW',
            lr0=0.001,
            cos_lr=True,
            mosaic=1.0,
            mixup=0.15,
            copy_paste=0.1,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            rect=False,
        )

if __name__ == '__main__':
    main()
