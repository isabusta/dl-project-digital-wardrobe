import os
import json
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from PIL import Image as PILImage
from torchvision import transforms
from torchvision.ops import box_iou, nms
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_processing import ClothingDatasetResize
from utility import collate_fn, plot_image_1
from attribute_data import TYPE_ORDER, LABEL_NAMES

# To do 
# Method for evaluation with different Score 
# Plot for res net with classification and box prediction 
# for training and for testing in same plot => store in json for later plot 
# Images with box detection and detection after filtering 
# have to train model again to create json file (weekend) ? 
# run evaluation and store in json file 
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

    # added by Isabelle — transform for the attribute model (same normalization as classifier)
    attribute_transformer = transforms.Compose([
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

    # sleeve and neckline attributes are not meaningful for bottoms
    BOTTOM_CATEGORIES = {"shorts", "trousers", "skirt"}
    BOTTOM_ATTRS      = {"texture", "fabric", "fit", "length"}

    # added by Isabelle — attr_model is optional, pipeline works without it
    def __init__(self, obj_detector, classifier, attr_model=None, debug=False, img_idx=0, eval_mode=False):
        self.device       = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.obj_detector = obj_detector.to(self.device)
        self.classifier   = classifier.to(self.device)
        self.attr_model   = attr_model.to(self.device) if attr_model is not None else None
        self.debug        = debug
        self.eval_mode    = eval_mode
        self.idx          = img_idx

    def detect_objects(self, img_tensor, score_threshold=0.5, iou_threshold=0.5):
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

        if len(boxes) == 0:
            return []

        # filter out low confidence boxes
        keep   = scores > score_threshold
        boxes  = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if len(boxes) == 0:
            return []

        # filter out boxes that are too similar
        # if labels are the same and boxes are to similar only keep one box 
        keep_indices = []
        for label in labels.unique():
            label_mask    = labels == label
            label_indices = label_mask.nonzero(as_tuple=True)[0]
            kept          = nms(boxes[label_mask], scores[label_mask], iou_threshold)
            keep_indices.append(label_indices[kept])

        keep_indices = torch.cat(keep_indices)
        boxes        = boxes[keep_indices]

        # plot boxes after filtering
        if self.debug:
            print(f"Boxes after filtering: {len(boxes)}")
            plot_image_1(img_tensor, {'boxes': boxes})

        return boxes.cpu().numpy()

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



    def predict_attributes(self, cropped_img):
        tensor = self.attribute_transformer(cropped_img).unsqueeze(0).to(self.device)
        self.attr_model.eval()
        with torch.inference_mode():
            logits = self.attr_model(tensor)
        return {t: LABEL_NAMES[t][logits[t].argmax(dim=1).item()] for t in TYPE_ORDER}

    def demo_image(self, image_path: str, score_threshold: float = 0.5, title: str = None):
        """Run full pipeline on one image and display a visualization."""
        COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']

        pil_img    = PILImage.open(image_path).convert("RGB")
        img_tensor = self.test_transform(pil_img)
        boxes      = self.detect_objects(img_tensor, score_threshold=score_threshold)

        if not len(boxes):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img_tensor.permute(1, 2, 0).numpy())
            ax.set_title("No items detected", fontsize=13)
            ax.axis("off")
            plt.tight_layout()
            plt.show()
            return

        crops = self.crop_img(img_tensor, boxes)
        n     = len(crops)

        items = []
        for box, crop in zip(boxes, crops):
            pred_id, pred_label = self.predict(crop)
            if self.attr_model is not None:
                attrs = self.predict_attributes(crop)
                if pred_label in self.BOTTOM_CATEGORIES:
                    attrs = {k: v for k, v in attrs.items() if k in self.BOTTOM_ATTRS}
            else:
                attrs = {}
            items.append({"box": box, "crop": crop, "label": pred_label, "attrs": attrs})

        fig = plt.figure(figsize=(12, max(4, n * 3.2)))
        gs  = gridspec.GridSpec(n, 3, width_ratios=[2, 1, 1.3], hspace=0.45, wspace=0.25)

        ax_main = fig.add_subplot(gs[:, 0])
        ax_main.imshow(img_tensor.permute(1, 2, 0).numpy())
        ax_main.set_title(title or os.path.basename(image_path),
                          fontsize=11, fontweight='bold')
        ax_main.axis("off")

        for i, item in enumerate(items):
            color           = COLORS[i % len(COLORS)]
            x1, y1, x2, y2 = item["box"]
            ax_main.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.5, edgecolor=color, facecolor='none'
            ))
            ax_main.text(
                x1 + 2, y1 + 13, f"#{i+1} {item['label']}",
                color='white', fontsize=7.5, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.88)
            )

            ax_crop = fig.add_subplot(gs[i, 1])
            ax_crop.imshow(item["crop"])
            ax_crop.set_title(f"#{i+1}  {item['label']}", fontsize=9,
                              fontweight='bold', color=color)
            ax_crop.axis("off")

            ax_attr = fig.add_subplot(gs[i, 2])
            ax_attr.axis("off")
            if item["attrs"]:
                lines = [f"{k:<10} {v}" for k, v in item["attrs"].items()]
                ax_attr.text(0.05, 0.92, "\n".join(lines), va='top', fontsize=9,
                             family='monospace', transform=ax_attr.transAxes,
                             linespacing=1.9)
            else:
                ax_attr.text(0.05, 0.5, "—", va='center', fontsize=9,
                             color='gray', transform=ax_attr.transAxes)

        plt.suptitle("Fashion Pipeline", fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.show()

    def evaluate(self, results, target):
        """
        Score 1: Exact Match:
            Only correct if detected == total GT boxes
            AND all predicted labels match GT labels (order independent)

        Score 2: Coverage:
            How many GT labels are covered by predictions (order independent)
            Penalized if more boxes detected than GT (precision penalty)
        """
        gt_labels      = target['labels']
        total          = len(gt_labels)
        detected       = len(results)

        gt_label_list  = [l.item() - 1 for l in gt_labels]
        pred_label_list = [r['category_id'] - 1 for r in results.values()]

        if detected == 0:
            return {
                "total":          total,
                "detected":       0,
                "exact_match":    0,
                "coverage":       0.0,
                "precision":      0.0,
            }

        # Exact Matches 
        # only valid if detected == total
        # count how many predicted labels are in GT (no duplicates)
        gt_remaining = gt_label_list.copy()
        matched = 0
        for pred in pred_label_list:
            if pred in gt_remaining:
                matched += 1
                gt_remaining.remove(pred)

        exact_match = 1 if (detected == total and matched == total) else 0

        # Coverage + Precision
        # coverage: how many GT labels are covered by predictions
        coverage  = matched / total if total > 0 else 0.0

        # precision: penalize if more boxes predicted than GT
        # if detected == total => precision = 1.0 
        # if detected > total  => precision < 1.0
        precision = total / detected if detected > total else 1.0

        return {
            "total":       total,
            "detected":    detected,
            "exact_match": exact_match,   # 1 for exact match, 0 otherwise
            "coverage":    round(coverage,  4),  # GT labels found
            "precision":   round(precision, 4),  # penalty for extra boxes
        }

    def run(self, test_data_path, output_dir='predictions', json_res = 'eval_results.json'):
        """
        Runs pipeline with ground truth to evaluate models and accuracy.
        Saves predictions as JSON in the same format as GT annotations.
        """
        os.makedirs(output_dir, exist_ok=True)

        # load test set
        test_dataset = ClothingDatasetResize(test_data_path, transform=self.test_transform)
        test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                  collate_fn=collate_fn, num_workers=2)

    
        all_exact    = 0
        all_coverage = 0.0
        all_precision = 0.0 
        all_images   = 0
        all_extra_boxes = 0

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
                    # added by Isabelle — attach attribute predictions if attr_model is available
                    # only show relevant attributes based on garment type
                    if self.attr_model is not None:
                        attrs = self.predict_attributes(crop)
                        if pred_label in self.BOTTOM_CATEGORIES:
                            attrs = {k: v for k, v in attrs.items() if k in self.BOTTOM_ATTRS}
                    else:
                        attrs = {}
                    results[f"item{i+1}"] = {
                        "bounding_box":  [float(x) for x in box],
                        "category_id":   pred_id + 1,
                        "category_name": pred_label,
                        "attributes":    attrs
                    }

            # evaluation against ground truth
            if self.eval_mode:
                scores        = self.evaluate(results, target)
                all_exact    += scores['exact_match']
                all_coverage += scores['coverage']
                all_precision += scores['precision']
                all_images   += 1
                extra_boxes = max(0, scores['detected'] - scores['total'])
                all_extra_boxes += extra_boxes

            # print result
            if self.debug and img_idx == self.idx:
                print(f"\nImage: {img_name}")
                if self.eval_mode:
                    print("── Ground Truth ──────────────────────")
                    for i, (gt_box, gt_label) in enumerate(zip(target['boxes'], target['labels'])):
                        print(f"  item{i+1}: {self.categories.get(gt_label.item()-1)} (id={gt_label.item()})")
                    print(f"── Scores ────────────────────────────")
                    print(f"  Exact Match: {scores['exact_match']} | detected {scores['detected']}/{scores['total']}")
                    print(f"  Coverage:    {scores['coverage']:.2%}")
                    print(f"  Precision:   {scores['precision']:.2%}")
                print("── Predictions ───────────────────────")
                for k, v in results.items():
                    print(f"  {k}: {v['category_name']} (id={v['category_id']})")
                break
            
            if not self.debug:
                # save JSON
                out_path = os.path.join(output_dir, img_name.replace('.jpg', '.json'))
                with open(out_path, 'w') as f:
                    json.dump(results, f, indent=4)

        if self.eval_mode and not self.debug:
            print(f"\nDone! {all_images} images")
            print(f"Exact Match:  {all_exact}/{all_images}  = {all_exact/all_images:.2%}")
            print(f"Avg Coverage: {all_coverage/all_images:.2%}")
            print(f"Avg Precision: {all_precision/all_images:.2%}") 
            print(f"Avg Extra Boxes: {all_extra_boxes/all_images:.2%}") 

            summary = {
                "total_images":    all_images,
                "exact_match":     all_exact,
                "exact_match_pct": round(all_exact    / all_images, 4) if all_images > 0 else 0,
                "avg_coverage":    round(all_coverage / all_images, 4) if all_images > 0 else 0,
                "avg_precision":   round(all_precision / all_images, 4) if all_images > 0 else 0,
                "avg_extra_boxes": round(all_extra_boxes / all_images, 4) if all_images > 0 else 0,
            }

            # save eval JSON
            eval_path = os.path.join(output_dir, json_res)
            with open(eval_path, 'w') as f:
                json.dump(summary, f, indent=4)