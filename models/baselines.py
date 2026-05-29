import numpy as np
import logging
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

class NaiveRandomWalk:
    """
    Naive Random Walk Baseline:
    Predicts that future surfaces will remain identical to the last observed surface.
    """
    def __init__(self, horizon: int = 1):
        self.horizon = horizon

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Naive model does not require training
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: Input tensor of shape (B, L, 1, M, N)
        Returns:
            Forecast of shape (B, h, M, N)
        """
        B, L, _, M, N = X.shape
        # Extract the last time step in the lookback window: shape (B, M, N)
        last_step = X[:, -1, 0, :, :]
        # Replicate it along the horizon dimension: shape (B, h, M, N)
        return np.repeat(last_step[:, np.newaxis, :, :], self.horizon, axis=1)


class HistoricalMean:
    """
    Historical Mean Baseline:
    Predicts that future surfaces will be the mean of the lookback window.
    """
    def __init__(self, horizon: int = 1):
        self.horizon = horizon

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: Input tensor of shape (B, L, 1, M, N)
        Returns:
            Forecast of shape (B, h, M, N)
        """
        # Calculate mean along the Lookback (L) dimension (axis=1)
        mean_surface = np.mean(X[:, :, 0, :, :], axis=1) # Shape: (B, M, N)
        return np.repeat(mean_surface[:, np.newaxis, :, :], self.horizon, axis=1)


class ExponentialSmoothing:
    """
    Single Exponential Smoothing Baseline:
    S_{t} = alpha * Y_{t} + (1 - alpha) * S_{t-1}
    Fitted grid-cell by grid-cell.
    """
    def __init__(self, horizon: int = 1, alpha: float = 0.3):
        self.horizon = horizon
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray):
        # We can optimize alpha, but using a standard baseline alpha=0.3 is common.
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: Input tensor of shape (B, L, 1, M, N)
        Returns:
            Forecast of shape (B, h, M, N)
        """
        B, L, _, M, N = X.shape
        predictions = np.zeros((B, self.horizon, M, N))
        
        for b in range(B):
            for i in range(M):
                for j in range(N):
                    # Time series for this cell
                    ts = X[b, :, 0, i, j]
                    # Compute exponential smoothing recursively
                    s = ts[0]
                    for t in range(1, L):
                        s = self.alpha * ts[t] + (1.0 - self.alpha) * s
                    
                    # For forecast horizon, the flat forecast is the last smoothed value
                    predictions[b, :, i, j] = s
                    
        return predictions


class GARCHModel:
    """
    GARCH(1,1) Baseline:
    Fits an independent GARCH(1,1) model per grid cell using the 'arch' package.
    Falls back to Historical Mean if GARCH fitting fails or the library is missing.
    """
    def __init__(self, horizon: int = 1):
        self.horizon = horizon
        self._has_arch = False
        try:
            import arch
            self._has_arch = True
        except ImportError:
            logger.warning("The 'arch' package is not installed. GARCH model will fall back to Historical Mean.")

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: Input tensor of shape (B, L, 1, M, N)
        Returns:
            Forecast of shape (B, h, M, N)
        """
        B, L, _, M, N = X.shape
        predictions = np.zeros((B, self.horizon, M, N))
        
        # If 'arch' package is not available, run Historical Mean fallback
        if not self._has_arch:
            fallback = HistoricalMean(horizon=self.horizon)
            return fallback.predict(X)
            
        from arch import arch_model
        
        for b in range(B):
            for i in range(M):
                for j in range(N):
                    ts = X[b, :, 0, i, j]
                    try:
                        # Rescale to prevent convergence errors
                        scale = 100.0
                        scaled_ts = ts * scale
                        
                        # Fit GARCH(1,1) (constant mean, GARCH(1,1) volatility)
                        # We use quiet mode to avoid spamming the log
                        model = arch_model(scaled_ts, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
                        res = model.fit(disp="off", show_warning=False)
                        
                        # Forecast h-steps ahead
                        forecasts = res.forecast(horizon=self.horizon, reindex=False)
                        # Extract forecasted conditional volatility
                        # GARCH forecasts variance, so we take sqrt and scale back
                        # 'variance' shape is (1, h)
                        cond_var = forecasts.variance.values[0]
                        forecast_iv = np.sqrt(np.clip(cond_var, 1e-6, None)) / scale
                        
                        predictions[b, :, i, j] = forecast_iv
                    except Exception as e:
                        # Fallback to mean of the local time series in case of fitting failure
                        predictions[b, :, i, j] = np.mean(ts)
                        
        return predictions


class HARRVModel:
    """
    HAR-RV (Heterogeneous Autoregressive) Baseline:
    Predicts future volatility using Daily, Weekly (5-day), and Monthly (20-day) averages.
    Model: σ_{t+1} = β_0 + β_d * σ_t^(1) + β_w * σ_t^(5) + β_m * σ_t^(20)
    We fit a linear regression per grid cell.
    """
    def __init__(self, horizon: int = 1):
        self.horizon = horizon
        self.regressors = {} # Stores fitted linear regressions per grid cell

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit HAR-RV linear coefficients.
        Args:
            X: Input tensor of shape (N_samples, L, 1, M, N)
            y: Target tensor of shape (N_samples, h, M, N)
        """
        N_samples, L, _, M, N = X.shape
        if L < 20:
            raise ValueError(f"HAR-RV requires lookback window L >= 20 to compute 20-day monthly averages. Got L={L}")
            
        for i in range(M):
            for j in range(N):
                # Prepare regression inputs: daily, weekly, monthly averages for all samples
                features = []
                targets = []
                
                for s in range(N_samples):
                    ts = X[s, :, 0, i, j] # Time series of length L
                    
                    # Compute components at the end of the window (index L-1)
                    val_d = ts[-1]
                    val_w = np.mean(ts[-5:])
                    val_m = np.mean(ts[-20:])
                    
                    features.append([val_d, val_w, val_m])
                    # Target: average of the target horizon (or we fit for each horizon step)
                    # We fit a multi-output regression if horizon > 1
                    targets.append(y[s, :, i, j])
                    
                reg = LinearRegression()
                reg.fit(np.array(features), np.array(targets))
                self.regressors[(i, j)] = reg

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: Input tensor of shape (B, L, 1, M, N)
        Returns:
            Forecast of shape (B, h, M, N)
        """
        B, L, _, M, N = X.shape
        predictions = np.zeros((B, self.horizon, M, N))
        
        for i in range(M):
            for j in range(N):
                features = []
                for b in range(B):
                    ts = X[b, :, 0, i, j]
                    val_d = ts[-1]
                    val_w = np.mean(ts[-5:])
                    val_m = np.mean(ts[-20:])
                    features.append([val_d, val_w, val_m])
                    
                reg = self.regressors.get((i, j))
                if reg is not None:
                    # Predict shape: (B, h)
                    pred = reg.predict(np.array(features))
                    # Reshape to (B, h) in case horizon=1 returns (B,)
                    if self.horizon == 1:
                        pred = pred[:, np.newaxis]
                    predictions[:, :, i, j] = pred
                else:
                    # Fallback to Random Walk if model not fitted
                    last_step = X[:, -1, 0, i, j]
                    predictions[:, :, i, j] = np.repeat(last_step[:, np.newaxis], self.horizon, axis=1)
                    
        # Clip to valid volatility bounds
        return np.clip(predictions, 0.01, 5.0)
