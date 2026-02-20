"""
Evaluation metrics for spatiotemporal extraction.

Based on standard NER and information extraction evaluation practices:
- Entity-level metrics (not token-level)
- Precision, Recall, F1 for extraction
- Normalization accuracy for temporal expressions
- Geocoding accuracy for spatial expressions

References:
- CoNLL-2003 NER evaluation
- TempEval-3 (TIMEX3 evaluation)
- SemEval'13 entity evaluation
- ACE TERN (Temporal Expression Recognition and Normalization)
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import re
from datetime import datetime
from geopy.distance import geodesic


@dataclass
class TemporalMetrics:
    """Metrics for temporal extraction evaluation"""

    # Extraction metrics (entity-level)
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    # Normalization metrics
    normalization_correct: int = 0
    normalization_total: int = 0

    # Per-type metrics (date, time, duration, set)
    type_correct: int = 0
    type_total: int = 0

    def precision(self) -> float:
        """Calculate precision: TP / (TP + FP)"""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    def recall(self) -> float:
        """Calculate recall: TP / (TP + FN)"""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    def f1_score(self) -> float:
        """Calculate F1: 2 * (P * R) / (P + R)"""
        p = self.precision()
        r = self.recall()
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    def normalization_accuracy(self) -> float:
        """Calculate normalization accuracy"""
        if self.normalization_total == 0:
            return 0.0
        return self.normalization_correct / self.normalization_total

    def type_accuracy(self) -> float:
        """Calculate type classification accuracy"""
        if self.type_total == 0:
            return 0.0
        return self.type_correct / self.type_total

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary"""
        return {
            "precision": round(self.precision(), 4),
            "recall": round(self.recall(), 4),
            "f1_score": round(self.f1_score(), 4),
            "normalization_accuracy": round(self.normalization_accuracy(), 4),
            "type_accuracy": round(self.type_accuracy(), 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


@dataclass
class SpatialMetrics:
    """Metrics for spatial extraction evaluation"""

    # Extraction metrics (entity-level)
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    # Geocoding metrics
    geocoding_attempted: int = 0
    geocoding_successful: int = 0

    # Coordinate accuracy (distance-based)
    distance_errors: List[float] = field(default_factory=list)  # in km

    # Location type accuracy
    type_correct: int = 0
    type_total: int = 0

    def precision(self) -> float:
        """Calculate precision: TP / (TP + FP)"""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    def recall(self) -> float:
        """Calculate recall: TP / (TP + FN)"""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    def f1_score(self) -> float:
        """Calculate F1: 2 * (P * R) / (P + R)"""
        p = self.precision()
        r = self.recall()
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    def geocoding_success_rate(self) -> float:
        """Calculate geocoding success rate"""
        if self.geocoding_attempted == 0:
            return 0.0
        return self.geocoding_successful / self.geocoding_attempted

    def mean_distance_error(self) -> float:
        """Calculate mean distance error in km"""
        if not self.distance_errors:
            return 0.0
        return sum(self.distance_errors) / len(self.distance_errors)

    def median_distance_error(self) -> float:
        """Calculate median distance error in km"""
        if not self.distance_errors:
            return 0.0
        sorted_errors = sorted(self.distance_errors)
        n = len(sorted_errors)
        if n % 2 == 0:
            return (sorted_errors[n//2 - 1] + sorted_errors[n//2]) / 2
        return sorted_errors[n//2]

    def accuracy_at_threshold(self, threshold_km: float = 25) -> float:
        """
        Calculate accuracy@k: percentage of predictions within k km of ground truth.
        Default: 25km (based on geoparsing literature)
        """
        if not self.distance_errors:
            return 0.0
        within_threshold = sum(1 for e in self.distance_errors if e <= threshold_km)
        return within_threshold / len(self.distance_errors)

    def accuracy_in_range(self, min_km: float, max_km: float) -> float:
        """
        Calculate accuracy within a distance range: percentage of predictions
        with distance errors between min_km and max_km (inclusive).
        """
        if not self.distance_errors:
            return 0.0
        in_range = sum(1 for e in self.distance_errors if min_km <= e <= max_km)
        return in_range / len(self.distance_errors)

    def type_accuracy(self) -> float:
        """Calculate location type classification accuracy"""
        if self.type_total == 0:
            return 0.0
        return self.type_correct / self.type_total

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary"""
        return {
            "precision": round(self.precision(), 4),
            "recall": round(self.recall(), 4),
            "f1_score": round(self.f1_score(), 4),
            "geocoding_success_rate": round(self.geocoding_success_rate(), 4),
            "mean_distance_error_km": round(self.mean_distance_error(), 2),
            "median_distance_error_km": round(self.median_distance_error(), 2),
            "percentage_within_25km": round(self.accuracy_at_threshold(25), 4),
            "percentage_25km_to_200km": round(self.accuracy_in_range(25.01, 200), 4),
            "percentage_above_200km": round(1.0 - self.accuracy_at_threshold(200), 4),
            "type_accuracy": round(self.type_accuracy(), 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


@dataclass
class OverallMetrics:
    """Combined metrics for spatiotemporal extraction"""

    temporal: TemporalMetrics = field(default_factory=TemporalMetrics)
    spatial: SpatialMetrics = field(default_factory=SpatialMetrics)

    # Overall system metrics
    total_documents: int = 0
    successful_extractions: int = 0
    total_processing_time: float = 0.0

    def success_rate(self) -> float:
        """Calculate extraction success rate"""
        if self.total_documents == 0:
            return 0.0
        return self.successful_extractions / self.total_documents

    def average_processing_time(self) -> float:
        """Calculate average processing time per document"""
        if self.total_documents == 0:
            return 0.0
        return self.total_processing_time / self.total_documents

    def combined_f1(self) -> float:
        """Calculate macro-averaged F1 across temporal and spatial"""
        temporal_f1 = self.temporal.f1_score()
        spatial_f1 = self.spatial.f1_score()
        return (temporal_f1 + spatial_f1) / 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert all metrics to dictionary"""
        return {
            "temporal": self.temporal.to_dict(),
            "spatial": self.spatial.to_dict(),
            "overall": {
                "combined_f1": round(self.combined_f1(), 4),
                "success_rate": round(self.success_rate(), 4),
                "average_processing_time_seconds": round(self.average_processing_time(), 2),
                "total_documents": self.total_documents,
            }
        }


def calculate_temporal_match(
    predicted: Dict[str, Any],
    ground_truth: Dict[str, Any],
    match_mode: str = "value_exact"
) -> bool:
    """
    Calculate if a predicted temporal entity matches ground truth.

    Args:
        predicted: Predicted temporal entity dict
        ground_truth: Ground truth temporal entity dict
        match_mode:
            - "text_exact": Exact match of raw text field
            - "text_fuzzy": Fuzzy match of raw text (word overlap >= 50%)
            - "value_exact": Exact match of normalized ISO 8601 values (RECOMMENDED)

    Returns:
        True if match, False otherwise
    """
    # Legacy support for old mode names
    if match_mode == "exact":
        match_mode = "text_exact"
    elif match_mode == "overlap":
        match_mode = "text_fuzzy"
    elif match_mode == "normalized":
        match_mode = "value_exact"

    if match_mode == "text_exact":
        return predicted.get("text", "").lower().strip() == ground_truth.get("text", "").lower().strip()

    elif match_mode == "text_fuzzy":
        pred_text = predicted.get("text", "").lower()
        gt_text = ground_truth.get("text", "").lower()
        # Check if there's substantial overlap (IoU-style)
        pred_words = set(pred_text.split())
        gt_words = set(gt_text.split())
        if not pred_words or not gt_words:
            return False
        intersection = len(pred_words & gt_words)
        union = len(pred_words | gt_words)
        return intersection / union >= 0.5

    elif match_mode == "value_exact":
        # Match based on normalized temporal value (RECOMMENDED for temporal evaluation)
        pred_norm = predicted.get("normalized", "")
        gt_norm = ground_truth.get("normalized", "")
        if not pred_norm or not gt_norm:
            return False
        # Normalize: strip T00:00:00 suffix so "2022-03-15T00:00:00" == "2022-03-15"
        def _strip_midnight(v: str) -> str:
            for suffix in ("T00:00:00", "T00:00"):
                if v.endswith(suffix):
                    return v[: -len(suffix)]
            return v
        return _strip_midnight(pred_norm) == _strip_midnight(gt_norm)

    return False


def calculate_spatial_match(
    predicted: Dict[str, Any],
    ground_truth: Dict[str, Any],
    match_mode: str = "fuzzy"
) -> Tuple[bool, Optional[float]]:
    """
    Calculate if a predicted spatial entity matches ground truth.

    Args:
        predicted: Predicted spatial entity dict
        ground_truth: Ground truth spatial entity dict
        match_mode:
            - "exact": Exact match of location text
            - "fuzzy": Fuzzy match (substring or word overlap >= 50%) (RECOMMENDED)

    Returns:
        Tuple of (is_match, distance_error_km)
    """
    # Text-based matching
    if match_mode == "exact":
        text_match = predicted.get("text", "").lower().strip() == ground_truth.get("text", "").lower().strip()
    else:  # fuzzy
        pred_text = predicted.get("text", "").lower()
        gt_text = ground_truth.get("text", "").lower()
        # Simple fuzzy match: check if one contains the other or significant overlap
        if pred_text in gt_text or gt_text in pred_text:
            text_match = True
        else:
            pred_words = set(pred_text.split())
            gt_words = set(gt_text.split())
            if not pred_words or not gt_words:
                text_match = False
            else:
                intersection = len(pred_words & gt_words)
                union = len(pred_words | gt_words)
                text_match = intersection / union >= 0.5

    # Calculate distance error if coordinates available
    distance_error = None
    if text_match and "latitude" in predicted and "latitude" in ground_truth:
        try:
            pred_coords = (predicted["latitude"], predicted["longitude"])
            gt_coords = (ground_truth["latitude"], ground_truth["longitude"])
            distance_error = geodesic(pred_coords, gt_coords).kilometers
        except:
            pass

    return text_match, distance_error


def normalize_temporal_value(value: str) -> str:
    """
    Normalize temporal value for comparison.
    Handles ISO 8601 variations.
    """
    if not value:
        return ""

    # Remove time zones, milliseconds for comparison
    value = re.sub(r'[+-]\d{2}:\d{2}$', '', value)  # Remove timezone
    value = re.sub(r'\.\d+', '', value)  # Remove milliseconds

    # Normalize date formats
    # 2022-03-15T00:00:00 -> 2022-03-15
    if 'T00:00:00' in value:
        value = value.replace('T00:00:00', '')

    return value.strip()


def compare_coordinates(
    pred_lat: float,
    pred_lon: float,
    gt_lat: float,
    gt_lon: float
) -> float:
    """
    Compare two coordinate pairs and return distance in km.

    Returns:
        Distance in kilometers
    """
    try:
        return geodesic((pred_lat, pred_lon), (gt_lat, gt_lon)).kilometers
    except:
        return float('inf')
