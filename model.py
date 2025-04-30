import os
from typing import Optional, Tuple, TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass

from transformers import BertConfig, BertModel

from data_preprocessing import DataPreprocessing
from data_processing import DataParameters, DataProcessing


@dataclass
class Bert4RecParams:
    vocab_size: int = 59049
    heads: int = 4
    num_hidden_layers: int = 4
    hidden_layer_size: int = 256
    num_pos: int = 20


class Bert4Rec(nn.Module):
    def __init__(self, params: Bert4RecParams):
        super().__init__()
        self.params = params

        self.bert_config = BertConfig(
            hidden_act="gelu",  # Hardcode gelu because bert4rec paper defines this
            vocab_size=params.vocab_size,
            num_attention_heads=params.heads,
            num_hidden_layers=params.num_hidden_layers,
            hidden_size=params.hidden_layer_size,
            max_position_embeddings=params.num_pos,
        )
        self.bert: BertModel = BertModel(self.bert_config)
        self.output = nn.Linear(params.hidden_layer_size, params.vocab_size)

    def forward(self, input_ids, attention_mask, output_logits=True):

        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        logits = self.output(bert_output.last_hidden_state)
        if output_logits:
            return logits
        else:
            return F.softmax(logits, -1)


@dataclass
class Bert4RecTrainingParams(Bert4RecParams):
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = (
        1e-3
    )
    weight_decay: float = 1e-4
    patience: int = 3
    scheduler_patience: int = 1
    scheduler_factor: float = 0.5


def train(data_processor: "DataProcessing", params: Bert4RecTrainingParams):
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
        verbose=True,
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
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, output_logits=True)
            loss = criterion(outputs.view(-1, params.vocab_size), labels.view(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            progress_bar.set_postfix({"train_loss": train_loss / train_batches})

        avg_train_loss = train_loss / train_batches

        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask, output_logits=True)
                loss = criterion(
                    outputs.view(-1, params.vocab_size), labels.view(-1)
                )
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{params.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            best_model_state = model.state_dict().copy()
            
            # Backup the new best model for later inference
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                },
                "best_bert4rec_model.pt",
            )
        else:
            early_stop_counter += 1
            if early_stop_counter >= params.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    
    # Test the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    test_loss = 0
    test_batches = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask, output_logits=True)
            loss = criterion(outputs.view(-1, params.vocab_size), labels.view(-1))

            # Update stats
            test_loss += loss.item()
            test_batches += 1

    avg_test_loss = test_loss / test_batches
    print(f"Test Loss: {avg_test_loss:.4f}")

    return model, avg_test_loss


if __name__ == "__main__":
    data_params = DataParameters(
        padding_token=0,
        masking_token=1,
    )
    data_preprocessor = DataPreprocessing(path="data/")
    data_processor = DataProcessing(preprocessor=data_preprocessor, params=data_params)

    training_params = Bert4RecTrainingParams(
        vocab_size=data_processor.get_token_count(),
        heads=8,
        num_hidden_layers=4,
        hidden_layer_size=256,
        num_pos=100,
        epochs=20,
        batch_size=32,
        learning_rate=1e-3,
    )

    model, test_loss = train(data_processor, training_params)
