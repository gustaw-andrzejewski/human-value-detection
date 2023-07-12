import numpy as np
import torch
import wandb
import yaml
from data.ArgumentsDataset import ArgumentsDataset
from models import GAT, GCN, GraphSAGE
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch_geometric.loader import DataLoader

# PATHS
CONFIG_PATH = "workspace/wandb_config.yaml"
BEST_MODEL_NAME = "best_model.pt"
DATASET_ROOT = "workspace/data/"

# CONSTANTS
MODEL = "model"
VAL_ACCURACY = "val_accuracy"
VAL_PRECISION = "val_precision"
VAL_RECALL = "val_recall"
VAL_F1 = "val_f1"
EPOCHS = 200
PATIENCE = 25


def train():
    with open(CONFIG_PATH, "r") as f:
        sweep_config = yaml.safe_load(f)

    wandb.init(project=sweep_config["project"])

    # instantiate dataset and dataloaders
    dataset = ArgumentsDataset(root=DATASET_ROOT, test_size=0.2)
    num_node_features = dataset[0].num_node_features
    num_classes = dataset[0].y.size(1)
    train_loader = DataLoader(dataset, batch_size=wandb.config.batch_size)

    # instantiate model
    if wandb.config.model == "GAT":
        model = GAT(
            num_node_features=num_node_features,
            num_classes=num_classes,
            num_hidden_units=wandb.config.num_hidden_units,
            num_heads=wandb.config.num_heads,
            dropout=wandb.config.dropout,
        )
    elif wandb.config.model == "GCN":
        model = GCN(
            num_node_features=num_node_features,
            num_classes=num_classes,
            num_hidden_units=wandb.config.num_hidden_units,
            dropout=wandb.config.dropout,
        )
    elif wandb.config.model == "GraphSAGE":
        model = GraphSAGE(
            num_node_features=num_node_features,
            num_classes=num_classes,
            num_hidden_units=wandb.config.num_hidden_units,
            dropout=wandb.config.dropout,
        )

    train_model(
        model,
        train_loader,
        EPOCHS,
        PATIENCE,
        wandb.config.learning_rate,
    )


def train_model(model, train_loader, epochs, patience, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = torch.compile(model)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Initialize some values for early stopping
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_loss, val_metrics = evaluate_model(model, train_loader, device, criterion)
        print(f"Epoch: {epoch+1}, train loss: {train_loss}, val loss: {val_loss}, val f1: {val_metrics[VAL_F1]}, val accuracy: {val_metrics[VAL_ACCURACY]}")
        # Update the learning rate
        scheduler.step()

        # Log the training and validation loss and metrics
        wandb.log(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_metrics[VAL_ACCURACY],
                "val_precision": val_metrics[VAL_PRECISION],
                "val_recall": val_metrics[VAL_RECALL],
                "val_f1": val_metrics[VAL_F1],
            }
        )

        # Check if early stopping condition is met
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset counter

            # # Save the model file
            # torch.save(model.state_dict(), BEST_MODEL_NAME)

            # # Log the model file as an artifact in wandb
            # artifact = wandb.Artifact(MODEL, type=MODEL)
            # artifact.add_file(BEST_MODEL_NAME)
            # wandb.run.log_artifact(artifact)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break


def train_one_epoch(model, loader, optimizer, device, criterion):
    model.train()
    epoch_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output[data.train_mask], data.y[data.train_mask].float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def evaluate_model(model, loader, device, criterion):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out[data.test_mask], data.y[data.test_mask].float())
            epoch_loss += loss.item()
            preds = out[data.test_mask].round().detach().cpu().numpy()
            labels = data.y[data.test_mask].cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="micro"
    )
    accuracy = accuracy_score(all_labels, all_preds)
    metrics = {
        VAL_ACCURACY: accuracy,
        VAL_PRECISION: precision,
        VAL_RECALL: recall,
        VAL_F1: fscore,
    }
    return epoch_loss / len(loader), metrics


if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as f:
        sweep_config = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=train)
