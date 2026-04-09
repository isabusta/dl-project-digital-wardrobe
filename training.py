import torch
from tqdm.auto import tqdm


def train_step_resnet(model: torch.nn.Module,
                      dataloader: torch.utils.data.DataLoader,
                      optimizer: torch.optim.Optimizer,
                      device="cuda"):
    # Put the model in train mode
    model.train()

    # Setup train loss and train
    total_train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):

        # X is the image, y the targets
        X = X.to(device)

        targets = []
        for t in y:
            d = {}
            d["boxes"] = (t["boxes"] * 224).to(device)
            d["labels"] = t["labels"].to(device)
            targets.append(d)

        # Forward pass
        # output will be a dictionary
        loss_dict = model(X, targets)

        # Calculate the loss
        train_loss_sum = sum(loss for loss in loss_dict.values())
        train_loss = train_loss_sum.item()

        # accumulate train loss
        total_train_loss += train_loss

        # optimizer zero grad
        optimizer.zero_grad()

        # loss backward
        train_loss_sum.backward()

        # optimizer step
        optimizer.step()

    # compute the average train loss


    average_train_loss = total_train_loss / len(dataloader)

    return average_train_loss


def test_step_resnet(model: torch.nn.Module,
                     dataloader: torch.utils.data.DataLoader,
                     device="cuda"):

    # 1. set model in evaluation mode
    model.eval()

    total_score = 0
    num_predictions = 0

    with torch.inference_mode():
        for images, targets in dataloader:

            # shift images to device
            images = [img.to(device) for img in images]

            # Forward Pass: create predictions for the category
            outputs = model(images)

            for i, output in enumerate(outputs):
                if len(output['scores']) > 0:
                    total_score += torch.mean(output['scores']).item()
                    num_predictions += 1

    # average confidence
    avg_confidence = total_score / num_predictions if num_predictions > 0 else 0

    return avg_confidence


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          device="cuda", epochs: int = 10):
    results = {"train_loss": [],
               "test_loss": []}

    for epoch in tqdm(range(epochs)):
        train_loss = train_step_resnet(model=model,
                                       dataloader=train_dataloader,
                                       optimizer=optimizer,
                                       device=device)

        test_loss = test_step_resnet(model=model,
                                     dataloader=test_dataloader,
                                     device=device)

        print(f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)

    return results
