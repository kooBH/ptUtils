"""
GradientStabilizer: Fix the Norm, Not the Gradient (ICML 2026, Huang et al.)

Optimizer-agnostic gradient transform that preserves the instantaneous
gradient direction while replacing its global L2 magnitude with a
statistically stabilized estimate rho_t = m_t / sqrt(v_t), where m_t / v_t
are EMAs of the gradient norm and squared norm (Algorithm 1 of the paper).

Usage (between backward() and optimizer.step()):

    stabilizer = GradientStabilizer(model.parameters())
    ...
    loss.backward()
    stabilizer.step()
    optimizer.step()
"""
import torch


class GradientStabilizer:
    """Rescales gradients in-place: g_tilde = (m_t / sqrt(v_t)) * g / ||g||_2.

    Args:
        parameters: iterable of Tensors or a torch.nn.Module.
        gamma1: EMA decay for the first moment of the gradient norm.
        gamma2: EMA decay for the second moment. Paper defaults: 0.6 / 0.999
                (fixed across all experiments in the paper).
        eps: numerical floor for divisions.

    Note: the paper initializes m_0 = v_0 = 0 with no bias correction; the
    uniform bound rho_t <= (1-gamma1)/sqrt(1-gamma2) * 1/sqrt(1-gamma1^2/gamma2)
    (Lemma 3.6) relies on this exact recursion, so it is kept as-is.
    Requires gamma1^2 < gamma2 for the bound to be well-defined.
    """

    def __init__(self, parameters, gamma1=0.6, gamma2=0.999, eps=1e-12):
        if isinstance(parameters, torch.nn.Module):
            parameters = parameters.parameters()
        self.params = [p for p in parameters if p.requires_grad]
        if not self.params:
            raise ValueError("GradientStabilizer::no parameters with requires_grad")
        if not (0.0 <= gamma1 < 1.0 and 0.0 <= gamma2 < 1.0):
            raise ValueError(f"GradientStabilizer::decays must be in [0,1): {gamma1}, {gamma2}")
        if gamma1 ** 2 >= gamma2:
            raise ValueError(f"GradientStabilizer::requires gamma1^2 < gamma2: {gamma1}, {gamma2}")
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.eps = eps
        self.m = 0.0  # EMA of ||g||_2
        self.v = 0.0  # EMA of ||g||_2^2

    @torch.no_grad()
    def step(self):
        """Stabilize current gradients in-place.

        Returns (rho, R): stabilized magnitude and raw gradient norm,
        for logging. Returns (R, R) unchanged when R is 0 or non-finite
        (EMA state is not updated in that case).
        """
        grads = [p.grad for p in self.params if p.grad is not None]
        if not grads:
            return 0.0, 0.0
        R = torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(g.detach()) for g in grads])
        ).item()
        if R <= 0.0 or not torch.isfinite(torch.tensor(R)):
            return R, R
        self.m = self.gamma1 * self.m + (1.0 - self.gamma1) * R
        self.v = self.gamma2 * self.v + (1.0 - self.gamma2) * R * R
        rho = self.m / (self.v ** 0.5 + self.eps)
        scale = rho / R
        torch._foreach_mul_(grads, scale)
        return rho, R

    def state_dict(self):
        return {"m": self.m, "v": self.v, "gamma1": self.gamma1, "gamma2": self.gamma2}

    def load_state_dict(self, state):
        self.m = state["m"]
        self.v = state["v"]
        self.gamma1 = state.get("gamma1", self.gamma1)
        self.gamma2 = state.get("gamma2", self.gamma2)
