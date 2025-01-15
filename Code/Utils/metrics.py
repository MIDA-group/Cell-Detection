
from typing_extensions import Self
import math
import numpy as np
import pandas as pd

def filter_edge_centroids(gt_centroids, pred_centroids, image_width, image_height, edge_threshold):
    """
    Filter out GT centroids and their predictions if GT is too close to image edges.

    Args:
        gt_centroids: list of (x, y) coordinates
        pred_centroids: list of (x, y) coordinates
        image_width: width of the image
        image_height: height of the image
        edge_threshold: minimum distance allowed from edges

    Returns:
        filtered_gt: GT centroids away from edges
        filtered_pred: corresponding predictions
        valid_indices: indices of valid GT centroids
    """

    # Check which GT centroids are far enough from edges
    valid_indices = []
    filtered_gt = []
    for i, (x, y) in enumerate(gt_centroids):
        if (x >= edge_threshold and x <= image_width - edge_threshold and
            y >= edge_threshold and y <= image_height - edge_threshold):
            valid_indices.append(i)
            filtered_gt.append((x, y))

    return filtered_gt,  valid_indices


class TotalLocalizationError:
    def __init__(self, images_info, alpha=0.5, slack=1, treshold=5, edge_threshold=10):
        """
        Initialize the Total Localization Error calculator.

        Parameters:
        -----------
        images_info : list of dict
            List of dictionaries, each containing:
            - 'gt_centroids': ground truth centroids DataFrame
            - 'pred_centroids': predicted centroids DataFrame
        alpha : float
            Penalty factor for false positives (extra predictions)
        slack : float
            Margin of error - Predicted centroids with distance <= slack from the gt centroids
            are considered a perfect match
        threshold : float
            Maximum accepted distance between a gt centroid and a predicted centroid.
            If the distance is greater than the threshold, the detection is considered a miss (FN + FP penalties added)
        """
        self.images_info = images_info
        self.alpha = alpha
        self.slack = slack
        self.threshold = treshold
        self.edge_threshold = edge_threshold

    def _relative_distance(self, gt_point, pred_point):
        """
        Calculate relative distance between ground truth and predicted centroids.

        Parameters:
        -----------
        gt_point : tuple
            Ground truth centroid coordinates (x, y)
        pred_point : tuple
            Predicted centroid coordinates (x, y)

        Returns:
        --------
        float: Relative distance
        """
        # Unpack coordinates
        gt_x, gt_y = gt_point
        pred_x, pred_y = pred_point

        # Handle virtual points with infinite coordinates
        if gt_x == float('inf') or pred_x == float('inf'):
            return 1.0

        # Calculate the euclidean distance
        distance = math.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
        return distance

    def _match_centroids(self, image_width, image_height, gt_points, pred_points, edge_threshold):
        """
        Match ground truth and predicted centroids to minimize total distance (Hungarian algorithm).

        Parameters:
        -----------
        gt_points : list
            Ground truth centroid coordinates
        pred_points : list
            Predicted centroid coordinates

        Returns:
        --------
        float: Localization error for the image
        dict: Additional error metrics
        """
        n_gt = len(gt_points)
        n_pred = len(pred_points)

        # Handle special cases
        if n_pred == 0:
            return float(n_gt), {'fn': n_gt, 'fp': 0, 'matches': 0, 'perfect_matches': 0, 'miss_detections': 0}, n_gt, 0
        if n_gt == 0:
            return float(n_pred * self.alpha), {'fn': 0, 'fp': n_pred, 'matches': 0, 'perfect_matches': 0, 'miss_detections': 0}, 1e-5, n_pred

        # Calculate base distance matrix
        distance_matrix = np.zeros((max(n_gt, n_pred), max(n_gt, n_pred)))

        # Fill the distance matrix with actual distances where possible
        for i in range(min(n_gt, distance_matrix.shape[0])):
            for j in range(min(n_pred, distance_matrix.shape[1])):
                distance = self._relative_distance(gt_points[i], pred_points[j])
                distance_matrix[i, j] = distance

        # Fill penalties for false negative
        if n_gt > n_pred:
            for i in range(n_gt):
                for j in range(n_pred, distance_matrix.shape[1]):
                    distance_matrix[i, j] = 10000

        # Use Hungarian algorithm for optimal matching
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        matched_distances = distance_matrix[row_ind, col_ind]

        filtered_gt, valid_indices = filter_edge_centroids(gt_points, pred_points, image_width, image_height, edge_threshold)

        # Calculate the error for each pairing
        errors = []
        for i in range(len(matched_distances)):
            if matched_distances[i] <= self.slack:
                errors.append(0)
            elif matched_distances[i] <= self.threshold:
                errors.append((matched_distances[i] - self.slack)/(self.threshold - self.slack))
            elif matched_distances[i] <= ((self.alpha+1)*self.threshold - self.alpha*self.slack):
                errors.append((matched_distances[i] - self.slack)/(self.threshold - self.slack))
            elif matched_distances[i] == 10000:
                errors.append(1)
            else:
                errors.append(1 + self.alpha)

        final_error = []
        for i in valid_indices:
            final_error.append(errors[i])
        miss_detections = np.count_nonzero(np.array(final_error) > 1)
        final_error.append(max(0, n_pred - n_gt)*self.alpha)
        perfect_matches = np.count_nonzero(np.array(final_error) == 0)

        fp =  max(0, n_pred - n_gt)
        n_gt = len(valid_indices)

        error_metrics = {
            'fn': max(0, n_gt - n_pred) + miss_detections,  # False negatives (missing detections)
            'fp': fp + miss_detections,  # False positives (extra predictions)
            'perfect_matches': perfect_matches,  # Matches within slack
            'miss_detections': miss_detections, # Detection over the threshold value
        }

        if len(valid_indices) == 0:
            return np.sum(final_error), error_metrics, 1e-5, n_pred
        else:
          tp = len(valid_indices)

        return np.sum(final_error), error_metrics, tp, n_pred

    def calculate_total_localization_error(self):
        """
        Calculate the average localization error across all images.

        Returns:
        --------
        float: average localization error
        list: Detailed error breakdown for each image
        """
        image_errors = []
        detailed_errors = []
        false_negatives = []
        false_positives = []
        true_positives = []

        for img_info in self.images_info:
            # Extract image information
            gt_centroids = img_info['gt_centroids']
            pred_centroids = img_info['pred_centroids']
            image_width = img_info['width']
            image_height = img_info['height']

            # Convert centroids to list of tuples
            gt_points = list(zip(gt_centroids['x'], gt_centroids['y']))
            pred_points = list(zip(pred_centroids['x'], pred_centroids['y']))

            # Calculate localization error for this image
            img_error, error_metrics, tp, n_pred = self._match_centroids(
                image_width, image_height, gt_points, pred_points, self.edge_threshold
            )

            fn = error_metrics['fn']
            fp = error_metrics['fp']
            precision =  tp / (tp + fp)
            recall = tp / (tp + fn)
            FScore = 2*tp / (2*tp + fn + fp)

            image_errors.append(img_error)
            false_negatives.append(fn)
            false_positives.append(fp)
            true_positives.append(tp)

            detailed_errors.append({
                'gt_count': tp,
                'pred_count': n_pred,
                'image_error': img_error,
                'false_negatives': fn,
                'false_positives': fp,
                'precision': precision,
                'recall': recall,
                'F-score': FScore,
                'perfect_matches': error_metrics['perfect_matches'],
                'miss_detections': error_metrics['miss_detections'],
            })

        # Calculate total localization error, precision, recall and Fscore
        total_error = np.sum(image_errors) / np.sum(true_positives)
        total_precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))
        total_recall = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))
        total_FScore = 2*np.sum(true_positives) / (2*np.sum(true_positives) + np.sum(false_negatives) + np.sum(false_positives))

        return total_error, total_precision, total_recall, total_FScore, detailed_errors, np.sum(true_positives)

import json 

def calculate_metrics(json_path,alpha):

    with open(json_path, 'r') as f:
        images_info = json.load(f)

    tle_calculator = TotalLocalizationError(
        images_info,
        alpha=alpha,
        slack=5,
        treshold=22,
        edge_threshold=25
    )

    total_error, total_precision, total_recall, total_FScore, detailed_errors, tp = tle_calculator.calculate_total_localization_error()
    file_name = 'evaluation_results_' + str(alpha) + '.txt'
    with open(file_name, 'w') as f:
        f.write(f"Total Localization Error: {total_error:.4f}\n")
        f.write(f"Precision: {total_precision:.4f}\n")
        f.write(f"Recall: {total_recall:.4f}\n")
        f.write(f"F-score: {total_FScore:.4f}\n")
        f.write("\nDetailed Errors:\n")

        for i, error in enumerate(detailed_errors, 1):
            f.write(f"\nImage {i}:\n")
            f.write(f"GT centroids: {error['gt_count']}\n")
            f.write(f"Pred centroids: {error['pred_count']}\n")
            f.write(f"Image error: {error['image_error']:.4f}\n")
            f.write(f"False negatives: {error['false_negatives']}\n")
            f.write(f"False positives: {error['false_positives']}\n")
            f.write(f"Precision: {error['precision']}\n")
            f.write(f"Recall: {error['recall']}\n")
            f.write(f"F-score: {error['F-score']}\n")
            f.write(f"Perfect matches (≤ Slack): {error['perfect_matches']}\n")
            f.write(f"Miss detections (> Threshold): {error['miss_detections']}\n")

    print("Results saved to ", file_name)

