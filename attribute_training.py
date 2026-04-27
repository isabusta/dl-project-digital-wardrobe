import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ExponentialLR

from attribute_data import TYPE_ORDER

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def train_step_attribute(model: torch.nn.Module,
                         dataloader: torch.utils.data.DataLoader,
                         optimizer: torch.optim.Optimizer,
                         device="mps"):

    model.train()

    criterion = nn.CrossEntropyLoss()
    total_train_loss = 0

    for batch, (X, y) in enumerate(dataloader):

        X = X.to(device)

        # Move each head's labels to device
        targets = {t: y[t].to(device) for t in TYPE_ORDER}

        # Forward pass — returns dict of logits per head
        outputs = model(X)

        # Sum CrossEntropyLoss across all 6 heads
        loss = sum(criterion(outputs[t], targets[t]) for t in TYPE_ORDER)

        total_train_loss += loss.item()

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = total_train_loss / len(dataloader)

    return average_train_loss


def val_step_attribute(model: torch.nn.Module,
                       dataloader: torch.utils.data.DataLoader,
                       device="mps"):

    model.eval()

    correct = {t: 0 for t in TYPE_ORDER}
    total = 0

    with torch.inference_mode():
        for images, y in dataloader:

            images = images.to(device)
            targets = {t: y[t].to(device) for t in TYPE_ORDER}

            outputs = model(images)

            total += images.size(0)

            for t in TYPE_ORDER:
                preds = outputs[t].argmax(dim=1)
                correct[t] += (preds == targets[t]).sum().item()

    # Per-head accuracy
    acc_per_head = {t: correct[t] / total for t in TYPE_ORDER}

    # Overall accuracy = mean across all heads
    avg_acc = sum(acc_per_head.values()) / len(TYPE_ORDER)

    return avg_acc, acc_per_head


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          device="mps",
          epochs: int = 10,
          scheduler: ExponentialLR = None):

    results = {
        "train_loss": [],
        "val_acc": [],
        "val_acc_per_head": [],
    }

    for epoch in tqdm(range(epochs)):

        train_loss = train_step_attribute(model=model,
                                         dataloader=train_dataloader,
                                         optimizer=optimizer,
                                         device=device)

        val_acc, acc_per_head = val_step_attribute(model=model,
                                                   dataloader=test_dataloader,
                                                   device=device)

        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              + " | ".join(f"{t}: {acc_per_head[t]:.4f}" for t in TYPE_ORDER))

        results["train_loss"].append(train_loss)
        results["val_acc"].append(val_acc)
        results["val_acc_per_head"].append(acc_per_head)

    return results
