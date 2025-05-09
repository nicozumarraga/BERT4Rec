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
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 10
    early_stop_warmup: int = 20
    early_stop_delta: float = 0.00001
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
# this is point-wise evaluation, we implement and use full ranking below.
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

            if "labels" in batch:
                labels = batch["labels"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
                total_loss += loss.item()
            else:
                # Just for predictions without calculating loss
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )

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

def full_rank_eval(
    model, loader, criterion, vocab_size: int, device="cuda", k=10
):
    model.eval()
    total_loss = 0
    batch_count = 0

    user_predictions = {}
    user_ground_truth = {}

    with torch.no_grad():
        for batch in loader:
            batch_count += 1
            user_ids = batch["user_id"].tolist()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            position_ids = batch["position_ids"].to(device)
            future_items = batch["future_items"]

            if "labels" in batch:
                labels = batch["labels"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
                total_loss += loss.item()
            else:
                # Just for predictions without calculating loss
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )

            # process for each user in batch the full ranking
            for i in range(input_ids.size(0)):
                user_id = user_ids[i]

                # reinitialize the user structures
                if user_id not in user_predictions:
                    user_predictions[user_id] = {}
                if user_id not in user_ground_truth:
                    user_ground_truth[user_id] = []

                # store ground truth future items (excluded padding tokens)
                if future_items is not None:
                    user_future = future_items[i].tolist()
                    user_future = [item for item in user_future if item != 0]
                    user_ground_truth[user_id].extend(user_future)

                # find the last valid position in the sequence and predict
                valid_indices = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]
                if len(valid_indices) == 0:
                    continue
                last_pos = valid_indices[-1].item()
                item_scores = outputs[i, last_pos].detach().cpu().numpy()

                # store predictions for all items (except padding token)
                for item_id, score in enumerate(item_scores):
                    if item_id == 0:
                        continue
                    user_predictions[user_id][item_id] = score

    # Calculate metrics for each user
    # TODO: move to specific recall and NDCG functions if this eval method works
    recall_scores = []
    ndcg_scores = []

    for user_id in user_predictions:
        ranked_items = sorted(
            user_predictions[user_id].keys(),
            key=lambda x: user_predictions[user_id][x],
            reverse=True
        )

        top_k_items = ranked_items[:k]

        # Recall
        ground_truth = user_ground_truth[user_id]
        hits = sum(1 for item in ground_truth if item in top_k_items)
        recall = hits / len(ground_truth) if ground_truth else 0
        recall_scores.append(recall)

        # NDCG
        dcg = 0
        for gt_item in ground_truth:
            if gt_item in top_k_items:
                rank = top_k_items.index(gt_item)
                dcg += 1 / np.log2(rank + 2)

        # Ideal DCG
        ideal_ranks = list(range(min(len(ground_truth), k)))
        idcg = sum(1 / np.log2(rank + 2) for rank in ideal_ranks)

        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)

    # Average across users
    avg_recall = np.mean(recall_scores) if recall_scores else 0
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
    avg_loss = total_loss / max(1, batch_count)

    return {
        "loss": avg_loss if batch_count > 0 else -avg_ndcg,
        f"recall@{k}": avg_recall,
        f"ndcg@{k}": avg_ndcg,
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
    best_ndcg = -np.inf
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

        val_results = full_rank_eval(
            model, val_loader, criterion, params.vocab_size, device, k=10
        )
        scheduler.step(val_results["loss"])

        print(
            f"Epoch {epoch+1}/{params.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_results['loss']:.4f}, Val recall@10: {val_results['recall@10']:.4f}, Val NDCG@10: {val_results['ndcg@10']:.4f}, LR: {scheduler.get_last_lr()}"
        )

        # Early stopping
        if val_results["ndcg@10"] > best_ndcg + params.early_stop_delta:
            best_ndcg = val_results["ndcg@10"]
            early_stop_counter = 0
            best_model_state = model.state_dict().copy()

            # Backup the new best model for later inference
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_ndcg@10": val_results["ndcg@10"],
                },
                "best_bert4rec_model.pt",
            )
        elif epoch >= params.early_stop_warmup:
            early_stop_counter += 1
            if early_stop_counter >= params.early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    model.load_state_dict(best_model_state)
    test_results = full_rank_eval(
        model, test_loader, criterion, params.vocab_size, device, k=10
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
