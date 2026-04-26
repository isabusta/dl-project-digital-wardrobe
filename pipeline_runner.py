import os
import json
import torch
from torchvision import transforms
from torchvision.ops import box_iou
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_processing import ClothingDatasetResize
from utility import collate_fn, plot_image

class PipelineRunner:

    train_transform_aug = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    classification_prediction_transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    categories = {
        0: "short sleeve top",    1: "long sleeve top",      2: "short sleeve outwear",
        3: "long sleeve outwear", 4: "vest",                  5: "sling",
        6: "shorts",              7: "trousers",              8: "skirt",
        9: "sleeve dress",        10: "long sleeve dress",   11: "vest dress",
        12: "sling dress"
    }

    def __init__(self, detector, classifier):
        self.device     = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector   = detector.to(self.device)
        self.classifier = classifier.to(self.device)

    def run(self, test_data_path, output_dir='predictions', debug=False):
        os.makedirs(output_dir, exist_ok=True)

        test_dataset = ClothingDatasetResize(test_data_path, transform=self.test_transform)
        test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                collate_fn=collate_fn, num_workers=2)

        self.detector.eval()
        self.classifier.eval()

        all_correct = 0
        all_total   = 0

        for img_idx, (images, targets) in enumerate(tqdm(test_loader, desc="Running Pipeline")):
            img_tensor = images[0].to(self.device)
            target     = targets[0]
            img_name   = test_dataset.images[img_idx]

            with torch.inference_mode():
                predictions = self.detector([img_tensor])

            _, all_boxes = find_best_clothing_box(predictions)

            pil_img      = transforms.ToPILImage()(img_tensor.cpu())
            gt_boxes     = target['boxes']
            gt_labels    = target['labels']
            pred_json    = {}
            no_detection = len(all_boxes) == 0

            for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                gt_label_0idx = gt_label.item() - 1
                item_key      = f"item{i+1}"

                if no_detection:
                    pred_json[item_key] = {
                        "gt_category_id":     gt_label.item(),
                        "gt_category_name":   self.categories.get(gt_label_0idx, 'unknown'),
                        "pred_category_id":   -1,
                        "pred_category_name": "No Detection",
                        "iou": 0.0
                    }
                    continue

                gt_box_tensor     = gt_box.unsqueeze(0)
                pred_boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
                ious              = box_iou(gt_box_tensor, pred_boxes_tensor)
                best_match_idx    = ious.argmax().item()
                best_box          = all_boxes[best_match_idx]

                x1, y1, x2, y2     = best_box
                cropped             = pil_img.crop((x1, y1, x2, y2))
                pred_id, pred_label = predict(
                    cropped,
                    self.classifier,
                    self.classification_prediction_transformer
                )

                pred_json[item_key] = {
                    "gt_category_id":     gt_label.item(),
                    "gt_category_name":   self.categories.get(gt_label_0idx, 'unknown'),
                    "pred_category_id":   pred_id + 1,
                    "pred_category_name": pred_label,
                    "iou":                round(ious[0, best_match_idx].item(), 4)
                }

                if pred_id == gt_label_0idx:
                    all_correct += 1
                all_total += 1

            # ── Debug: Plot + Print nach erstem Bild abbrechen ──
            if debug:
                print(f"\nBild: {img_name}")
                print("── Ground Truth ──────────────────────")
                for k, v in pred_json.items():
                    print(f"  {k}: {v['gt_category_name']} (id={v['gt_category_id']})")
                print("── Predictions ───────────────────────")
                for k, v in pred_json.items():
                    print(f"  {k}: {v['pred_category_name']} (id={v['pred_category_id']}) | IoU={v['iou']}")
                plot_image(images[0], target)
                break  # nach erstem Bild stoppen

            # ── Normal: JSON speichern ───────────────────────────
            out_path = os.path.join(output_dir, img_name.replace('.jpg', '.json'))
            with open(out_path, 'w') as f:
                json.dump(pred_json, f, indent=4)

        # ── Evaluation (wird bei debug=True übersprungen) ────────
        if not debug:
            accuracy = all_correct / all_total if all_total > 0 else 0
            print(f"\nFertig! {all_total} Items aus {len(test_dataset)} Bildern")
            print(f"Accuracy: {all_correct}/{all_total} = {accuracy:.2%}")
            return accuracy