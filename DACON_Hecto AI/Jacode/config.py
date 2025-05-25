import torch

CFG = {
    'IMG_SIZE': 224,
    'BATCH_SIZE': 32,
    'EPOCHS': 20,
    'LEARNING_RATE': 1e-4,
    'SEED': 42
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 