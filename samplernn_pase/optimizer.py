from typing import Callable, Optional
import torch.nn.functional as F
import torch.optim


class AdamClipped(torch.optim.Adam):

    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        def closure_wrapper():
            for group in self.param_groups:
                for p in group['params']:
                    F.hardtanh(p.grad, -1, 1, inplace=True)
            return None
        return super().step(closure_wrapper)
