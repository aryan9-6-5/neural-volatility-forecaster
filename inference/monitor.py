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
        
    def log_prediction(self, y_true: np.ndarray, y_pred: np.ndarray, baseline_surface: np.ndarray) -> dict:
        """
        Record a new prediction event, compute metrics, and check for drift.
        Args:
            y_true: True volatility surface grid (M, N)
            y_pred: Predicted volatility surface grid (M, N)
            baseline_surface: Average historical training surface grid (M, N)
        Returns:
            status_dict: current drift metrics and retrain flags
        """
        # 1. Compute prediction RMSE
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        self.recent_rmses.append(rmse)
        if len(self.recent_rmses) > self.rolling_window_days:
            self.recent_rmses.pop(0)
            
        # 2. Compute data drift (KL divergence of true surface vs baseline)
        kl = calculate_kl_divergence(y_true, baseline_surface)
        self.recent_kls.append(kl)
        if len(self.recent_kls) > self.rolling_window_days:
            self.recent_kls.pop(0)
            
        # 3. Calculate rolling statistics
        rolling_rmse = float(np.mean(self.recent_rmses))
        rolling_kl = float(np.mean(self.recent_kls))
        
        # 4. Check thresholds
        performance_drift = rolling_rmse > (self.baseline_rmse * self.rmse_threshold_factor)
        data_drift = rolling_kl > self.kl_threshold
        
        trigger_retrain = performance_drift or data_drift
        
        status = {
            "current_rmse": rmse,
            "rolling_rmse": rolling_rmse,
            "current_kl_divergence": kl,
            "rolling_kl_divergence": rolling_kl,
            "performance_drift_detected": bool(performance_drift),
            "data_drift_detected": bool(data_drift),
            "trigger_retrain": bool(trigger_retrain)
        }
        
        if trigger_retrain:
            logger.warning(
                f"DRIFT DETECTED: rolling_rmse={rolling_rmse:.4f} (baseline={self.baseline_rmse:.4f}), "
                f"rolling_kl={rolling_kl:.4f} (limit={self.kl_threshold:.4f}). Retraining triggered!"
            )
            
        return status
