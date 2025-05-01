from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

from data_preprocessing import DataPreprocessing
from data_processing import DataParameters, DataProcessing
from model import Bert4Rec, Bert4RecParams


@dataclass
class Bert4RecTrainingParams(Bert4RecParams):
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 3
    early_stop_warmup: int = 20
    early_stop_delta: float = 0.001
    scheduler_patience: int = 1
    scheduler_factor: float = 0.5


def get_top_k_items(outputs: np.ndarray, k: int = 10):
    return np.argsort(outputs, axis=1)[:, -k:]


def compute_recall_at_k(top_k_outputs: np.ndarray, all_labels: np.ndarray):
    hits = sum(1 for idx, label in enumerate(all_labels) if label in top_k_outputs[idx])

    return hits / len(all_labels)


def compute_ndcg_at_k(top_k_outputs: np.ndarray, all_labels: np.ndarray):
    total_ndcg_score = 0
    for i, label in enumerate(all_labels):
        rank_matches = np.where(top_k_outputs[i] == label)[0]
        if rank_matches:
            ndcg = 1 / np.log2(rank_matches[0] + 2)  # +2 because rank is 0-indexed
        else:
            ndcg = 0

        total_ndcg_score += ndcg

    return total_ndcg_score / len(all_labels)


# TODO: implement this in the training loop
def evaluate(model, loader, criterion, vocab_size: int, device="cuda", k=10):
    model.eval()
    total_loss = 0
    batch_count = 0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch_count += 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            position_ids = batch["position_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
            total_loss += loss.item()

            for i in range(input_ids.size(0)):
                for j in range(input_ids.size(1)):
                    if labels[i, j] != 0:
                        preds = outputs[i, j].detach().cpu().numpy()
                        label = labels[i, j].item()

                        all_outputs.append(preds)
                        all_labels.append(label)

    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    all_top_10_outputs = get_top_k_items(all_outputs, k=10)

    # Compute metrics
    recall = compute_recall_at_k(all_top_10_outputs, all_labels)
    ndcg = compute_ndcg_at_k(all_top_10_outputs, all_labels)

    # Compute average loss
    avg_loss = total_loss / batch_count

    return {
        "loss": avg_loss,
        f"recall@{k}": recall,
        f"ndcg@{k}": ndcg,
    }


def train(data_processor: DataProcessing, params: Bert4RecTrainingParams):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device.type}")

    model = Bert4Rec(params).to(device)

    # Ignore 0 tokens because this is either padding or a token that's also in the input sequence
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = optim.Adam(
        model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay
    )

    # Learning rate schedular
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=params.scheduler_factor,
        patience=params.scheduler_patience,
    )

    # Early stopping metrics
    best_val_loss = float("inf")
    early_stop_counter = 0
    best_model_state = None

    train_loader, val_loader, test_loader = data_processor.get_dataloaders(
        batch_size=params.batch_size
    )

    for epoch in range(params.epochs):

        model.train()
        train_loss = 0
        train_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{params.epochs}")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            position_ids = batch["position_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            loss = criterion(outputs.view(-1, params.vocab_size), labels.view(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            progress_bar.set_postfix({"train_loss": train_loss / train_batches})

        avg_train_loss = train_loss / train_batches

        val_results = evaluate(
            model, val_loader, criterion, params.vocab_size, device, k=10
        )
        scheduler.step(val_results["loss"])

        print(
            f"Epoch {epoch+1}/{params.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_results['loss']:.4f}, LR: {scheduler.get_last_lr()}"
        )

        # Early stopping
        if val_results["loss"] < best_val_loss - params.early_stop_delta:
            best_val_loss = val_results["loss"]
            early_stop_counter = 0
            best_model_state = model.state_dict().copy()

            # Backup the new best model for later inference
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_results["loss"],
                },
                "best_bert4rec_model.pt",
            )
        elif epoch >= params.early_stop_warmup:
            early_stop_counter += 1
            if early_stop_counter >= params.early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    model.load_state_dict(best_model_state)
    test_results = evaluate(
        model, val_loader, criterion, params.vocab_size, device, k=10
    )
    print(f"Test Loss: {test_results['loss']:.4f}")

    return test_results, val_results, epoch


# Testing the model
if __name__ == "__main__":
    data_params = DataParameters(
        padding_token=0,
        masking_token=1,
    )
    data_preprocessor = DataPreprocessing(path="data/")
    data_processor = DataProcessing(preprocessor=data_preprocessor, params=data_params)

    training_params = Bert4RecTrainingParams(
        vocab_size=data_processor.get_token_count(),
        heads=2,
        num_hidden_layers=1,
        hidden_layer_size=64,
        num_pos=1500,
        epochs=1,
        batch_size=128,
        learning_rate=1e-3,
    )

    last_test_results, last_val_results, epoch = train(data_processor, training_params)

    print(last_test_results)
