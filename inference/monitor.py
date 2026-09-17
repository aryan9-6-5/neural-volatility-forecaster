import numpy as np
import logging

logger = logging.getLogger(__name__)

def normalize_to_density(surface: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Normalize a volatility surface (M x N) to sum to 1.0,
    forming a spatial probability density function for KL divergence.
    """
    # Ensure positive values
    surf_clamped = np.clip(surface, epsilon, None)
    total_sum = np.sum(surf_clamped)
    if total_sum <= 0:
        return np.ones_like(surface) / surface.size
    return surf_clamped / total_sum

def calculate_kl_divergence(P_surface: np.ndarray, Q_surface: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute Kullback-Leibler (KL) Divergence between two normalized surface distributions.
    P_surface: Incoming current surface (actual or predicted)
    Q_surface: Baseline training surface
    """
    P = normalize_to_density(P_surface, epsilon=epsilon)
    Q = normalize_to_density(Q_surface, epsilon=epsilon)

    # KL Divergence formula: sum(P * ln(P / Q))
    kl = np.sum(P * np.log(P / Q))
    return float(kl)

class DriftMonitor:
    """
    Monitor data drift (KL Divergence) and forecast performance drift (RMSE).
    Accumulates errors in a rolling window to flag retraining needs.

    Data drift and performance drift are logged independently, since they
    become available at different times: data drift can be checked the moment
    a new snapshot arrives, but performance drift requires a forecast made in
    advance to be compared against the real outcome once it's later observed
    (see inference/prediction_store.py — never against itself).
    """
    def __init__(self, rolling_window_days: int = 10, kl_threshold: float = 0.5, rmse_threshold_factor: float = 1.5):
        self.rolling_window_days = rolling_window_days
        self.kl_threshold = kl_threshold
        self.rmse_threshold_factor = rmse_threshold_factor

        # History buffers
        self.recent_rmses = []
        self.recent_kls = []
        self.baseline_rmse = 0.02 # default baseline, can be updated dynamically

    def update_baseline_rmse(self, rmse: float):
        """Update baseline validation RMSE from training metrics."""
        self.baseline_rmse = rmse
        logger.info(f"Drift monitor baseline RMSE updated to {self.baseline_rmse:.4f}")

    def log_data_drift(self, current_surface: np.ndarray, baseline_surface: np.ndarray) -> dict:
        """Record the KL divergence of an incoming surface against the training baseline."""
        kl = calculate_kl_divergence(current_surface, baseline_surface)
        self.recent_kls.append(kl)
        if len(self.recent_kls) > self.rolling_window_days:
            self.recent_kls.pop(0)
        return {"current_kl_divergence": kl}

    def log_performance(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Record a matured forecast's real error: y_pred must be a prediction made
        in advance (e.g. from PredictionStore), and y_true the later-observed
        real surface for that same target date — never the same snapshot.
        """
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        self.recent_rmses.append(rmse)
        if len(self.recent_rmses) > self.rolling_window_days:
            self.recent_rmses.pop(0)
        return {"current_rmse": rmse}

    def get_status(self) -> dict:
        """
        Single source of truth for current drift status, used by both the
        /predict response and the /monitor/drift endpoint.
        """
        rolling_rmse = float(np.mean(self.recent_rmses)) if self.recent_rmses else 0.0
        rolling_kl = float(np.mean(self.recent_kls)) if self.recent_kls else 0.0

        # No matured predictions yet means we simply can't assess performance drift.
        performance_drift = bool(self.recent_rmses) and rolling_rmse > (self.baseline_rmse * self.rmse_threshold_factor)
        data_drift = bool(self.recent_kls) and rolling_kl > self.kl_threshold
        trigger_retrain = performance_drift or data_drift

        status = {
            "rolling_rmse": rolling_rmse,
            "rolling_kl_divergence": rolling_kl,
            "recent_rmses": list(self.recent_rmses),
            "recent_kls": list(self.recent_kls),
            "baseline_rmse": self.baseline_rmse,
            "kl_threshold": self.kl_threshold,
            "rmse_threshold_factor": self.rmse_threshold_factor,
            "performance_drift_detected": performance_drift,
            "data_drift_detected": data_drift,
            "trigger_retrain": trigger_retrain,
        }

        if trigger_retrain:
            logger.warning(
                f"DRIFT DETECTED: rolling_rmse={rolling_rmse:.4f} (baseline={self.baseline_rmse:.4f}), "
                f"rolling_kl={rolling_kl:.4f} (limit={self.kl_threshold:.4f}). Retraining triggered!"
            )

        return status
