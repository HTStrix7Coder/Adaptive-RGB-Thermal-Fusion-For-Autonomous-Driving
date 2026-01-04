import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Models.dataset import create_dataloaders

train_loader, _ = create_dataloaders(4, 0, (512, 640))

if train_loader is not None:
    try:
        batch = next(iter(train_loader))
        print(batch['rgb'].shape)
        print(batch['thermal'].shape)
        print(batch['name'])
    except Exception as e:
        print(f"Error iterating over train_loader: {e}")
else:
    print("train_loader is None, cannot fetch batch.")

