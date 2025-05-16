import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def calculate_IOU(boxA, boxB):
    """
    Compute Intersection over Union between two boxes.
    Args:
        boxA: List[float], [x1, y1, x2, y2] coordinates of box A
        boxB: List[float], [x1, y1, x2, y2] coordinates of box B
        top-left (x1, y1) and bottom-right (x2,y2) 
        normalized with respect to frame size, where (0.0, 0.0) corresponds to the top left, and (1.0, 1.0) corresponds to bottom right.
    """
    x_left_inarea = max(boxA[0], boxB[0])
    y_top_inarea = max(boxA[1], boxB[1])
    x_right_inarea = min(boxA[2], boxB[2])
    y_bottom_inarea = min(boxA[3], boxB[3])

    in_area = max(0, x_right_inarea - x_left_inarea) * max(0, y_bottom_inarea - y_top_inarea)
    if in_area == 0:
        return 0.0

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = in_area / float(boxA_area + boxB_area - in_area)
    return iou

def compute_accuracy(true_label, pred_label):
    """
    Compute classification accuracy
    """
    correct_num = (pred_label == true_label).sum()
    return correct_num / len(true_label)


def compute_PR(true_label, pred_score):
    sorted_indices = np.argsort(-np.array(pred_score))# sort in descending order
    y_true = np.array(true_label)[sorted_indices]

    tp = 0
    fp = 0
    precision = []
    recall = []
    total_positives = np.sum(y_true == 1) + 1e-10  # Avoid division by zero

    for i in range(len(y_true)):
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        prec = tp / (tp + fp)
        recall = tp / total_positives
        precision.append(prec)
        recall.append(recall)
    return np.array(precision), np.array(recall)

def compute_ap(precision, recall): #copy from talknet
  """Compute Average Precision according to the definition in VOCdevkit.
  Precision is modified to ensure that it does not decrease as recall
  decrease.
  Args:
    precision: A float [N, 1] numpy array of precisions
    recall: A float [N, 1] numpy array of recalls
  Raises:
    ValueError: if the input is not of the correct format
  Returns:
    average_precison: The area under the precision recall curve. NaN if
      precision and recall are None.
  """
  if precision is None:
    if recall is not None:
      raise ValueError("If precision is None, recall must also be None")
    return np.NAN

  if not isinstance(precision, np.ndarray) or not isinstance(
      recall, np.ndarray):
    raise ValueError("precision and recall must be numpy array")
  if precision.dtype != float or recall.dtype != float:
    raise ValueError("input must be float numpy array.")
  if len(precision) != len(recall):
    raise ValueError("precision and recall must be of the same size.")
  if not precision.size:
    return 0.0
  if np.amin(precision) < 0 or np.amax(precision) > 1:
    raise ValueError("Precision must be in the range of [0, 1].")
  if np.amin(recall) < 0 or np.amax(recall) > 1:
    raise ValueError("recall must be in the range of [0, 1].")
  if not all(recall[i] <= recall[i + 1] for i in range(len(recall) - 1)):
    raise ValueError("recall must be a non-decreasing array")

  recall = np.concatenate([[0], recall, [1]])
  precision = np.concatenate([[0], precision, [0]])

  # Smooth precision to be monotonically decreasing.
  for i in range(len(precision) - 2, -1, -1):
    precision[i] = np.maximum(precision[i], precision[i + 1])

  indices = np.where(recall[1:] != recall[:-1])[0] + 1
  average_precision = np.sum(
      (recall[indices] - recall[indices - 1]) * precision[indices])
  return average_precision


def compute_auc(y_true, y_score):
    return np.trapz(y_true, y_score)# modify later

def evaluate_detection(det_boxes, det_labels, det_scores,
                                true_boxes, true_labels, true_difficulties,threshold=0.5,
                                    visualize=False, output_dir="outputs_plots"):
    """
    Evaluate detection performance using IoU for area 
    label here should be label_id, not label name
    mAP, AUC, and Accuracy for scores
    filter difficult ones
    visualize: bool, whether to visualize the results
    """
    all_true = []
    all_pred_score = []
    iou_list = []

    for i in tqdm(range(len(true_labels)), desc="Evaluating"):
        pred_box = det_boxes[i]
        pred_score = det_scores[i]
        true_box = true_boxes[i]
        true_label = true_labels[i]
        difficulty = true_difficulties[i]
        
        if difficulty ==1:
            continue  # skip difficult
        iou = calculate_IOU(pred_box, true_box)
        iou_list.append(iou)
        all_pred_score.append(pred_score)
        all_true.append(true_label)
    
    all_pred_label = [1 if s >= threshold else 0 for s in all_pred_score]

    # PR
    precision, recall = compute_PR(all_true, all_pred_score)
    ap = compute_ap(precision, recall)

    # AUC
    

    # Accuracy
    accuracy = compute_accuracy(np.array(all_true), np.array(all_pred_label))

    # IoU
    avg_iou = np.mean(iou_list)

    # PR curve plot
    if visualize:
        os.makedirs(output_dir, exist_ok=True)
        plt.figure()
        plt.plot(recall, precision, label=f'PR Curve (AP={ap:.2f})')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "pr_curve.png"))
        plt.close()

    return ap, accuracy, avg_iou
