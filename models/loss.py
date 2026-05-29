import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Standard default grid expiries
DEFAULT_GRID_TAUS = np.array([1/52, 2/52, 1/12, 2/12, 3/12, 6/12, 1.0])

class SmoothnessRegularizedLoss(nn.Module):
    """
    Composite training loss for volatility surface forecasting:
    L_total = L_MSE + lambda_strike * L_strike + lambda_expiry * L_expiry + lambda_cal * L_calendar + lambda_but * L_butterfly
    """
    def __init__(
        self, 
        lambda_strike: float = 0.01, 
        lambda_expiry: float = 0.01,
        lambda_calendar: float = 0.0,
        lambda_butterfly: float = 0.0,
        grid_taus: np.ndarray = None
    ):
        super().__init__()
        self.lambda_strike = lambda_strike
        self.lambda_expiry = lambda_expiry
        self.lambda_calendar = lambda_calendar
        self.lambda_butterfly = lambda_butterfly
        
        # Keep grid taus as a tensor for calendar spread penalty calculation
        taus = grid_taus if grid_taus is not None else DEFAULT_GRID_TAUS
        self.register_buffer("grid_taus_tensor", torch.from_numpy(taus).float())

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> tuple:
        """
        Args:
            y_pred: Predicted forecast of shape (B, h, M, N)
            y_true: True surface of shape (B, h, M, N)
        Returns:
            total_loss: scalar tensor for optimization
            metrics_dict: dict of float values for training logs
        """
        # 1. Reconstruction MSE Loss
        mse_loss = F.mse_loss(y_pred, y_true)

        # 2. Strike smoothness (dim=-2) with reflection padding
        # Pad top and bottom of dim=-2
        padded_strike = F.pad(y_pred, pad=(0, 0, 1, 1), mode="reflect")
        strike_d2 = padded_strike[:, :, 2:, :] - 2 * padded_strike[:, :, 1:-1, :] + padded_strike[:, :, :-2, :]
        strike_loss = torch.mean(strike_d2 ** 2)

        # 3. Expiry smoothness (dim=-1) with reflection padding
        # Pad left and right of dim=-1
        padded_expiry = F.pad(y_pred, pad=(1, 1, 0, 0), mode="reflect")
        expiry_d2 = padded_expiry[:, :, :, 2:] - 2 * padded_expiry[:, :, :, 1:-1] + padded_expiry[:, :, :, :-2]
        expiry_loss = torch.mean(expiry_d2 ** 2)

        # 4. (Optional) Calendar spread arbitrage penalty
        calendar_loss = torch.tensor(0.0, device=y_pred.device)
        if self.lambda_calendar > 0:
            # Total variance w = IV^2 * tau
            # self.grid_taus_tensor shape is (N,) -> Reshape to (1, 1, 1, N)
            taus = self.grid_taus_tensor.view(1, 1, 1, -1)
            w = (y_pred ** 2) * taus
            # w must be non-decreasing in tau: w_j <= w_{j+1}
            # Penalty if w_j > w_{j+1} -> (w_j - w_{j+1}) > 0
            w_diff = w[:, :, :, :-1] - w[:, :, :, 1:]
            calendar_loss = torch.mean(torch.relu(w_diff) ** 2)

        # 5. (Optional) Butterfly spread smile convexity penalty
        butterfly_loss = torch.tensor(0.0, device=y_pred.device)
        if self.lambda_butterfly > 0:
            # SMILE must be convex along strike dim (dim -2): d2 >= 0
            # Penalty if d2 < 0 -> -d2 > 0
            d2 = y_pred[:, :, 2:, :] - 2 * y_pred[:, :, 1:-1, :] + y_pred[:, :, :-2, :]
            butterfly_loss = torch.mean(torch.relu(-d2) ** 2)

        # Combined loss
        total_loss = (
            mse_loss
            + self.lambda_strike * strike_loss
            + self.lambda_expiry * expiry_loss
            + self.lambda_calendar * calendar_loss
            + self.lambda_butterfly * butterfly_loss
        )

        metrics = {
            "mse_loss": mse_loss.item(),
            "strike_loss": strike_loss.item(),
            "expiry_loss": expiry_loss.item(),
            "calendar_loss": calendar_loss.item(),
            "butterfly_loss": butterfly_loss.item(),
            "total_loss": total_loss.item()
        }

        return total_loss, metrics
