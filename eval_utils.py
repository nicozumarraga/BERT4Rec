import torch
import torch.nn as nn
import numpy as np

# TODO: implement properly and test
def compute_recall_at_k(batch_outputs: torch.Tensor, batch_labels: torch.Tensor, k=10):
    batch_outputs = batch_outputs.detach().cpu().numpy()
    batch_labels = batch_labels.detach().cpu().numpy()
    
    batch_labels = batch_labels.reshape(-1) 
    top_k_items = np.argsort(batch_outputs, axis=1)[:, -k:]
    
    # Check if true label is in top-k predictions
    hits = 0
    for i, label in enumerate(batch_labels):
        if label in top_k_items[i]:
            hits += 1
    
    # Calculate recall@k
    recall = hits / len(batch_labels)
    
    return recall


# TODO: implement properly and test
def compute_ndcg_at_k(batch_outputs: torch.Tensor, batch_labels: torch.Tensor, k=10):
    batch_outputs = batch_outputs.detach().cpu().numpy()
    batch_labels = batch_labels.detach().cpu().numpy()
    
    batch_labels = batch_labels.reshape(-1)    
    top_k_indices = np.argsort(batch_outputs, axis=1)[:, -k:]    
    top_k_indices = np.flip(top_k_indices, axis=1)
    
    # Calculate NDCG for each sample
    ndcg_scores = []
    for i, label in enumerate(batch_labels):
        # Check if true label is in top-k predictions
        if label in top_k_indices[i]:
            # Find position of true label in top-k (0-indexed)
            rank = np.where(top_k_indices[i] == label)[0][0]
            # Calculate DCG
            ndcg = 1 / np.log2(rank + 2)  # +2 because rank is 0-indexed
        else:
            ndcg = 0
        
        ndcg_scores.append(ndcg)
    
    # Calculate average NDCG@k
    ndcg = np.mean(ndcg_scores)
    
    return ndcg


# TODO: implement this in the training loop
def evaluate(model, loader, criterion, vocab_size: int, device="cuda", k=10):
    model.eval()
    total_loss = 0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)            
            
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
            total_loss += loss.item()
            
    # TODO: compute this
    all_outputs = torch.stack(all_outputs)
    all_labels = torch.tensor(all_labels)
    
    # Compute metrics
    recall = compute_recall_at_k(all_outputs, all_labels, k=k)
    ndcg = compute_ndcg_at_k(all_outputs, all_labels, k=k)
    
    # Compute average loss
    avg_loss = total_loss / len(all_labels) if all_labels else 0
    
    return {
        "loss": avg_loss,
        f"recall@{k}": recall,
        f"ndcg@{k}": ndcg,
    }