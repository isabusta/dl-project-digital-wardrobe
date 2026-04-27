import os
import json
import torch
from torchvision import transforms
from torchvision.ops import box_iou, nms
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_processing import ClothingDatasetResize
from utility import collate_fn, plot_image_1

class Pipeline:

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

    def __init__(self, obj_detector, classifier, debug=False, eval_mode=True):
        self.device       = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.obj_detector = obj_detector.to(self.device)
        self.classifier   = classifier.to(self.device)
        self.debug        = debug
        self.eval_mode = eval_mode

    def detect_objects(self, img_tensor, score_threshold=0.5, iou_threshold=0.4):
        # model to eval mode
        self.obj_detector.eval()

        with torch.inference_mode():
            predictions = self.obj_detector([img_tensor.to(self.device)])

        boxes  = predictions[0]['boxes']
        scores = predictions[0]['scores']
        labels = predictions[0]['labels']

        # plot boxes before filtering
        if self.debug:
            print(f"Boxes before filtering: {len(boxes)}")
            plot_image_1(img_tensor, predictions[0])

        keep   = scores > score_threshold
        boxes  = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if len(boxes) == 0:
            return []

        # filter out boxes that are too similar
        # to prevent returning multiple boxes for same object
        keep_indices = []
        for label in labels.unique():
            label_mask    = labels == label
            label_indices = label_mask.nonzero(as_tuple=True)[0]
            kept          = nms(boxes[label_mask], scores[label_mask], iou_threshold)
            keep_indices.append(label_indices[kept])

        keep_indices = torch.cat(keep_indices)

        # plot boxes after filtering
        if self.debug:
            print(f"Boxes after filtering: {len(keep_indices)}")
            plot_image_1(img_tensor, {'boxes': boxes[keep_indices]})

        return boxes[keep_indices].cpu().numpy()

    def crop_img(self, img_tensor, boxes):
        pil_img = transforms.ToPILImage()(img_tensor.cpu())
        return [pil_img.crop((x1, y1, x2, y2)) for x1, y1, x2, y2 in boxes]

    def predict(self, cropped_img):
        # create tensor of cropped image
        tensor = self.classification_prediction_transformer(cropped_img).unsqueeze(0).to(self.device)

        # set model to eval mode
        self.classifier.eval()

        with torch.inference_mode():
            # compute y logits
            logits = self.classifier(tensor)

        probs      = torch.softmax(logits, dim=1)
        pred_id    = torch.argmax(probs, dim=1).item()
        pred_label = self.categories[pred_id]

        return pred_id, pred_label

    def evaluate(self, results, target):
        """
        Evaluates predictions against ground truth.
        Returns: all_correct, all_total
        """
        gt_boxes     = target['boxes']
        gt_labels    = target['labels']
        pred_boxes_t = torch.tensor([r['bounding_box'] for r in results.values()
                                    if r['bounding_box']], dtype=torch.float32)

        correct = 0
        total   = 0

        for gt_box, gt_label in zip(gt_boxes, gt_labels):
            gt_label_0idx = gt_label.item() - 1

            if len(pred_boxes_t) == 0:
                total += 1
                continue

            ious           = box_iou(gt_box.unsqueeze(0), pred_boxes_t)
            best_match_idx = ious.argmax().item()
            matched_item   = list(results.values())[best_match_idx]

            if matched_item['category_id'] - 1 == gt_label_0idx:
                correct += 1
            total += 1

        return correct, total

    def run(self, test_data_path, output_dir='predictions'):
        """
        Runs pipeline with ground truth to evaluate models and accuracy.
        Saves predictions as JSON in the same format as GT annotations.
        """
        os.makedirs(output_dir, exist_ok=True)

        # load test set
        test_dataset = ClothingDatasetResize(test_data_path, transform=self.test_transform)
        test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                  collate_fn=collate_fn, num_workers=2)

        all_correct = 0
        all_total   = 0

        for img_idx, (images, targets) in enumerate(tqdm(test_loader, desc="Running Pipeline")):
            img_tensor = images[0].to(self.device)
            target     = targets[0]
            img_name   = test_dataset.images[img_idx]

            # object detection
            predicted_boxes = self.detect_objects(img_tensor)

            # crop and classify
            results = {}
            if len(predicted_boxes) == 0:
                # No object detected
                results = {}   
            else:
                crops = self.crop_img(img_tensor, predicted_boxes)
                for i, (box, crop) in enumerate(zip(predicted_boxes, crops)):
                    pred_id, pred_label = self.predict(crop)
                    results[f"item{i+1}"] = {
                        "bounding_box":  [float(x) for x in box],
                        "category_id":   pred_id + 1,
                        "category_name": pred_label
                    }

            # evaluation against ground truth
            if self.eval_mode:
                correct, total = self.evaluate(results, target)
                all_correct   += correct
                all_total     += total

            # print result
            if self.debug:
                print(f"\nImage: {img_name}")
                if self.eval_mode:
                    print("── Ground Truth ──────────────────────")
                    for i, (gt_box, gt_label) in enumerate(zip(target['boxes'], target['labels'])):
                        print(f"  item{i+1}: {self.categories.get(gt_label.item()-1)} (id={gt_label.item()})")
                print("── Predictions ───────────────────────")
                for k, v in results.items():
                    print(f"  {k}: {v['category_name']} (id={v['category_id']})")
                break

            # save JSON
            out_path = os.path.join(output_dir, img_name.replace('.jpg', '.json'))
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=4)

        if self.eval_mode and not self.debug:
            accuracy = all_correct / all_total if all_total > 0 else 0
            print(f"\nDone! {all_total} items from {len(test_dataset)} images")
            print(f"Accuracy: {all_correct}/{all_total} = {accuracy:.2%}")
            return accuracy