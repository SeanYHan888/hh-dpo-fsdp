import torch


# warmup iteration
class WarmupQuantileAccumulator:
    """
    Accumulate margins during warmup and estimate initial quantile threshold (tau_0).
    We store warmup margins (optionally winsorized) and compute:
      tau_0 = quantile(margins, q)
    where typically q = 1 - delta, so that P(M >= tau) ~ delta.

    """

    def __init__(self, q):
        self.q = q
        self._buf: list[torch.Tensor] = []

    # margins: Tensor([batch_size]) for this warmup step
    @torch.no_grad()
    def update(self, batch_margins: torch.Tensor):
        t = batch_margins.detach().float().view(-1)
        if t.numel() == 0:
            return
        self._buf.append(t.cpu())

    def finalize(self):
        if len(self._buf) == 0:
            return 0.0
        all_m = torch.cat(self._buf, dim=0)
        tau0 = torch.quantile(all_m, self.q).item()
        return float(tau0)


# EMA
class EMAUpdate:
    def __init__(self, tau_0, q, momentum):
        self.tau = tau_0
        self.q = q
        self.lam = momentum

    # threshold tau equations
    def update_tau(self, batch_margins: torch.Tensor):
        t = batch_margins.detach().float().view(-1)
        if t.numel() == 0:
            return self.tau
        batch_tau = torch.quantile(t, self.q).item()
        self.tau = (1.0 - self.lam) * self.tau + self.lam * batch_tau
        return self.tau
