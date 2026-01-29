import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def grad(outputs, inputs):
    """Helper function to compute gradients"""
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True
    )[0]

class PINN(nn.Module):
    """
    Physics-Informed Neural Network with configurable layers
    Input: 6 parameters (par1, par2, par3, par4, x, t)
    Output: 2 parameters (vx, h)
    """
    def __init__(self, n_layers=4, hidden_dim=50):
        """
        Args:
            n_layers: Number of hidden layers (excluding input and output layers)
            hidden_dim: Dimension of hidden layers
        """
        super(PINN, self).__init__()
        
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # Input layer: 7 inputs
        layers = [nn.Linear(7, hidden_dim), nn.Tanh()]
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        # Output layer: 6 outputs (vx, vy, sxx, sxy, syy, h)
        layers.append(nn.Linear(hidden_dim, 6))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, inputs):
        """
        Forward pass through the network
        
        Args:
            inputs: Tensor containing concatenated physics parameters (par1, par2, par3, par4), spatial coordinate, and time
            
        Returns:
            outputs: Tuple of predicted (vx, vy, sxx, sxy, syy, h)
        """        
        # Pass through network
        output = self.network(inputs)
                
        return output

    def data_loss(self, y_pred, y_true):
        """Data fitting loss (MSE)"""
        loss = torch.mean((y_pred[:,-1] - y_true[:,-1])**2)
        return loss
    
    def physics_loss(self, X_phys):
        """
        Physics-informed loss based on governing equations
        This is a mock structure - replace with actual physics equations
        """ 

        # Forward pass
        # X shape: (batch_size, 7) -> par1, par2, par3, par4, x, y, t
        # Y shape: (batch_size, 6) -> vx, vy, sxx, sxy, syy, h
        par1 = X_phys[:, 0:1]
        par2 = X_phys[:, 1:2]
        par3 = X_phys[:, 2:3]
        par4 = X_phys[:, 3:4]
        x = X_phys[:, 4:5].requires_grad_(True)
        y = X_phys[:, 5:6].requires_grad_(True)
        t = X_phys[:, 6:7].requires_grad_(True)
        
        # Reconstruct input with gradient-enabled x, y, t
        X_phys_grad = torch.cat([par1, par2, par3, par4, x, y, t], dim=1)
        
        # Forward pass with gradients
        y_pred = self.network(X_phys_grad)
        
        vx = y_pred[:, 0:1]
        vy = y_pred[:, 1:2]
        sxx = y_pred[:, 2:3]
        sxy = y_pred[:, 3:4]
        syy = y_pred[:, 4:5]
        h = y_pred[:, 5:6]

        # First-order gradients
        dh_dt = grad(h, t)
        dh_dx = grad(h, x)
        dh_dy = grad(h, y)

        dsxx_dt = grad(sxx, t)
        dsxx_dx = grad(sxx, x)
        dsxx_dy = grad(sxx, y)

        dsxy_dt = grad(sxy, t)
        dsxy_dx = grad(sxy, x)
        dsxy_dy = grad(sxy, y)
        dsyx_dx = dsxy_dx
        dsyx_dy = dsxy_dy

        dsyy_dt = grad(syy, t)
        dsyy_dx = grad(syy, x)
        dsyy_dy = grad(syy, y)

        # Second-order gradients
        ddsxx_dxx = grad(dsxx_dx, x)
        ddsxx_dxy = grad(dsxx_dx, y)
        ddsxx_dyx = ddsxx_dxy
        ddsxx_dyy = grad(dsxx_dy, y)

        ddsxy_dxx = grad(dsxy_dx, x)
        ddsyx_dxx = ddsxy_dxx
        ddsxy_dxy = grad(dsxy_dx, y)
        ddsxy_dyx = ddsxy_dxy
        ddsyx_dxy = ddsxy_dxy
        ddsyx_dyx = ddsxy_dxy
        ddsxy_dyy = grad(dsxy_dy, y)
        ddsyx_dyy = ddsxy_dyy

        ddsyy_dxx = grad(dsyy_dx, x)
        ddsyy_dxy = grad(dsyy_dx, y)
        ddsyy_dyx = ddsyy_dxy
        ddsyy_dyy = grad(dsyy_dy, y)

        alpha1 = par1
        alpha2 = par3
        alpha3 = par2

        # v_j = alpha3 d_dx_i s_ij
        loss_eq11_x = ( vx - alpha3 * ( dsxx_dx + dsxy_dy ) )**2
        loss_eq11_y = ( vy - alpha3 * ( dsyx_dx + dsyy_dy ) )**2

        # dh_dt = -alpha3 d_dx_j (h d_dx_i s_ij) + 1 - h
        loss_eq10 = ( 
                        dh_dt \
                        + alpha3 * ( dh_dx * dsxx_dx + dh_dx * dsyx_dy + dh_dy * dsxy_dx + dh_dy * dsyy_dy ) \
                        + alpha3 * h * ( ddsxx_dxx + ddsyx_dyx + ddsxy_dxy + ddsyy_dyy ) \
                        - 1 + h 
                    )**2

        # (1 + alpha2 (d_dt + v_k d_dx_k))(s_ij-h delta_ij) = alpha1^2 h (dd_dxx_ik s_kj + dd_dxx_jk s_ki + 2 dd_dxx_km s_mk delta_ij)
        loss_eq9_xx = (
                        ( sxx - h ) + alpha2*(dsxx_dt - dh_dt ) + alpha2*( vx * dsxx_dx + vy * dsxx_dy - vx * dh_dx - vy * dh_dy ) \
                        - alpha1**2 * h * ( 
                            2 * ddsxx_dxx + 2 * ddsyx_dxy + 2 * ( ddsxx_dxx + ddsyy_dyy + ddsxy_dyx + ddsyx_dxy) 
                        )
                      )**2
        loss_eq9_yy = (
                        ( syy - h ) + alpha2*( dsyy_dt - dh_dt ) + alpha2*( vx * dsyy_dx + vy * dsyy_dy - vx * dh_dx - vy * dh_dy ) \
                        - alpha1**2 * h * ( 
                            2 * ddsyy_dyy + 2 * ddsxy_dyx + 2 * ( ddsxx_dxx + ddsyy_dyy + ddsxy_dyx + ddsyx_dxy) 
                        )
                      )**2
        loss_eq9_xy = (
                        ( sxy ) + alpha2*( dsxy_dt ) + alpha2*( vx * dsxy_dx + vy * dsxy_dy ) \
                        - alpha1**2 * h * ( 
                            ddsxy_dxx + ddsyy_dxy + ddsxx_dyx + ddsxy_dyy
                        )
                      )**2
        
        return ( loss_eq10 + loss_eq11_x + loss_eq11_y + loss_eq9_xx + loss_eq9_yy + loss_eq9_xy ).mean()

class PINNTrainer:
    """Trainer class for PINN with optional LBFGS refinement."""
    
    def __init__(self, model, learning_rate=1e-4, weight_decay=1e-6,
                 lbfgs_max_iter=500, clip_grad_norm=1.0):
        self.model = model
        self.clip_grad_norm = clip_grad_norm

        self.optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.1, patience=200, min_lr=1e-6
        )

        self.lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=lbfgs_max_iter,
            history_size=50,
            line_search_fn="strong_wolfe"
        )

    def _compute_loss(self, X_data, y_data, X_phys, lambda_data=1.0, lambda_physics=1.0):
        """Compute total loss and components (extend here when you add physics)."""
        
        # Data loss
        y_data_pred = self.model(X_data)
        loss_data = self.model.data_loss(y_data_pred, y_data)

        # Physics loss
        loss_phys = self.model.physics_loss(X_phys)
        
        #Total loss
        loss = lambda_data * loss_data + lambda_physics * loss_phys

        return loss, loss_data, loss_phys

    def train_step(self, X_data, y_data, X_phys, lambda_data=1.0, lambda_physics=1.0, refine=False):
        """
        One training step.
        - refine=False: AdamW step
        - refine=True: LBFGS step (full-batch recommended)
        """
        if not refine:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            loss, loss_data, loss_phys = self._compute_loss(X_data, y_data, X_phys, lambda_data, lambda_physics)
            loss.backward()

            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            self.optimizer.step()

            return {
                "total_loss": float(loss.detach().cpu()),
                "data_loss": float(loss_data.detach().cpu()),
                "physics_loss": float(loss_phys.detach().cpu()),
            }

        # ---- LBFGS refinement ----
        self.model.train()

        # Note: LBFGS will call this multiple times.
        def closure():
            self.lbfgs.zero_grad(set_to_none=True)
            loss, loss_data, loss_phys = self._compute_loss(X_data, y_data, X_phys, lambda_data, lambda_physics)
            loss.backward()
            return loss

        loss, loss_data, loss_phys = self._compute_loss(X_data, y_data, X_phys, lambda_data, lambda_physics)
        self.lbfgs.step(closure)
        loss_after, loss_data_after, loss_phys_after = self._compute_loss(X_data, y_data, X_phys, lambda_data, lambda_physics)

        return {
            "total_loss": float(loss_after.detach().cpu()),
            "data_loss": float(loss_data_after.detach().cpu()),
            "physics_loss": float(loss_phys_after.detach().cpu())
        }
    
    def test_step(self, X, y_true):
        """
        Single test step
        
        Args:
            par1-par4: Physics parameters (batch_size, 1)
            x, y, t: Coordinates (batch_size, 1)
            vx_true, h_true: Ground truth values (optional, for supervised learning)
        """
        with torch.no_grad():
            # Forward pass
            y_predict = self.model(X)
            
            # Compute losses
            loss_data = self.model.data_loss(y_predict, y_true)
            
            # Physics loss
            # loss_phys = self.model.physics_loss(par1, par2, par3, par4, x, y, t)
            
        return {
            'data_loss': loss_data.item() if isinstance(loss_data, torch.Tensor) else 0.0,
            # 'physics_loss': loss_phys.item()
        }