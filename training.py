import torch
from tqdm.auto import tqdm


def train_step_resnet(model: torch.nn.Module,
                      dataloader: torch.utils.data.DataLoader,
                      optimizer: torch.optim.Optimizer,
                      device="cuda"):
    # Put the model in train mode
    model.train()

    # Setup train loss and train
    total_train_loss, total_box_loss, train_acc = 0, 0, 0

    for batch, (X, y) in enumerate(dataloader):

        # X is the image, y the targets
        X = X.to(device)

        targets = []
        for t in y:
            d = {}
            d["boxes"] = (t["boxes"]).to(device)
            d["labels"] = t["labels"].to(device)
            targets.append(d)

        # Forward pass
        # output will be a dictionary
        loss_dict = model(X, targets)

        # Calculate the loss
        train_loss_sum = sum(loss for loss in loss_dict.values())
        train_loss_boxes_sum = loss_dict['loss_box_reg'] + loss_dict['loss_rpn_box_reg']
        train_loss = train_loss_sum.item()
        train_loss_boxes = train_loss_boxes_sum.item()


        # accumulate train loss
        total_train_loss += train_loss
        total_box_loss += train_loss_boxes

        # optimizer zero grad
        optimizer.zero_grad()

        # loss backward
        train_loss_sum.backward()

        # optimizer step
        optimizer.step()

    # compute the average train loss


    average_train_loss = total_train_loss / len(dataloader)
    average_train_loss_boxes = total_box_loss / len(dataloader)

    return average_train_loss, average_train_loss_boxes


def test_step_resnet(model: torch.nn.Module,
                     dataloader: torch.utils.data.DataLoader,
                     device="cuda"):

    # 1. set model in evaluation mode
    model.eval()

    total_score = 0
    num_predictions = 0
    total_boxes, total_gt_boxes = 0, 0
    score_threshold = 0.5

    with torch.inference_mode():
        for images, targets in dataloader:

            # shift images to device
            images = [img.to(device) for img in images]

            # Forward Pass: create predictions for the category
            outputs = model(images)

            for output, target in zip(outputs, targets):
                scores = output['scores']
                scores = scores[scores > score_threshhold]
                if len(scores) > 0:
                    total_score += torch.mean(scores).item()
                    num_predictions += 1

                pred_boxes = output['boxes']
                gt_boxes = target['boxes'].to(device)
                total_gt_boxes += len(gt_boxes)
                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    iou = box_iou(pred_boxes, gt_boxes)
                    max_iou, _ = iou.max(dim=0)
                    total_boxes += (max_iou >= 0.5).sum().item()

    # average confidence
    avg_confidence = total_score / num_predictions if num_predictions > 0 else 0
    recall = total_boxes / total_gt_boxes if total_gt_boxes > 0 else 0

    return avg_confidence, recall


from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ExponentialLR


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          device=device, epochs: int = 10,
          scheduler: ExponentialLR = None):
    results = {
        "train_loss": [],
        'train_box_loss': [], 
        "test_confidence": [],
        'test_recall': []
        
    }

    for epoch in tqdm(range(epochs)):
        train_loss, box_loss  = train_step_resnet(model=model,
                                       dataloader=train_dataloader,
                                       optimizer=optimizer,
                                       device=device)

        confidence, recall = test_step_resnet(model=model,
                                     dataloader=test_dataloader,
                                     device=device)
        if scheduler is not None:
            scheduler.step()

        print(f'Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Box Loss: {box_loss:.4f} | Confidence: {confidence:.4f} | Recall: {recall:.4f}')

        results["train_loss"].append(train_loss)
        results['train_box_loss'].append(box_loss)
        results["test_confidence"].append(confidence)
        results['test_recall'].append(recall)

    return results
