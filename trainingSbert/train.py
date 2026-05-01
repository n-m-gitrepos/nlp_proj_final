import torch
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader

def build_model(model_name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def build_dataloader(examples, batch_size: int):
    return DataLoader(examples, shuffle=True, batch_size=batch_size)

def train_model(model, dataloader, loss_name: str, epochs: int):

    if loss_name == "cosine":
        train_loss = losses.CosineSimilarityLoss(model)
    elif loss_name == "multiple_negatives":
        train_loss = losses.MultipleNegativesRankingLoss(model)
    else:
        raise ValueError("Unsupported loss")
        
    model.fit(train_objectives=[(dataloader, train_loss)], epochs=epochs, show_progress_bar=True)

def save_model_pt(model, path: str):
    torch.save(model, path)