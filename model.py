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
    num_hidden_layers: int = 4  # TODO: use params from bert4rec
    hidden_layer_size: int = 256  # TODO: use params from bert4rec
    emb_dim: Tuple[int, ...] = (256,)  # TODO: implement
    num_pos: int = 128


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
        1e-3  # Changed from 0.05 to a more appropriate value for Adam
    )
    weight_decay: float = 1e-4
    patience: int = 3  # For early stopping
    scheduler_patience: int = 1  # For LR scheduler
    scheduler_factor: float = 0.5


def train(data_processor: "DataProcessing", params: Bert4RecTrainingParams):
    # Get dataloaders
    train_loader, val_loader, test_loader = data_processor.get_dataloaders(
        batch_size=params.batch_size
    )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = Bert4Rec(params).to(device)

    # Define loss function (ignoring padding tokens - assuming 0 is padding)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Initialize optimizer with weight decay for regularization
    optimizer = optim.Adam(
        model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay
    )

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=params.scheduler_factor,
        patience=params.scheduler_patience,
        verbose=True,
    )

    # Early stopping variables
    best_val_loss = float("inf")
    early_stop_counter = 0
    best_model_state = None

    # Training loop
    for epoch in range(params.epochs):

        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{params.epochs}")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["labels"].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids, attention_mask, output_logits=True)

            # Calculate loss
            # Reshape outputs and targets for cross-entropy loss
            loss = criterion(outputs.view(-1, params.vocab_size), target_ids.view(-1))

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update stats
            train_loss += loss.item()
            train_batches += 1

            # Update progress bar
            progress_bar.set_postfix({"train_loss": train_loss / train_batches})

        avg_train_loss = train_loss / train_batches

        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target_ids = batch["labels"].to(device)

                # Forward pass
                outputs = model(input_ids, attention_mask, output_logits=True)

                # Calculate loss
                loss = criterion(
                    outputs.view(-1, params.vocab_size), target_ids.view(-1)
                )

                # Update stats
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)

        # Print epoch stats
        print(
            f"Epoch {epoch+1}/{params.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            best_model_state = model.state_dict().copy()
            # Save best model
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

    # Load best model for evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Evaluate on test set
    model.eval()
    test_loss = 0
    test_batches = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask, output_logits=True)

            # Calculate loss
            loss = criterion(outputs.view(-1, params.vocab_size), target_ids.view(-1))

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
