from __future__ import annotations

import torch


RELEASE_NAMES = {
    "R0_GEOMETRY_ONLY": [],
    "R1_O": ["o_raw"],
    "R2_C": ["sh_coeffs"],
    "R3_V": ["v_raw"],
    "R4_O_C": ["o_raw", "sh_coeffs"],
    "R5_O_V": ["o_raw", "v_raw"],
    "R6_C_V": ["sh_coeffs", "v_raw"],
    "R7_O_C_V_FULL": ["o_raw", "sh_coeffs", "v_raw"],
}


class GaussianState(torch.nn.Module):
    def __init__(self, n: int = 4096, release: str = "R0_GEOMETRY_ONLY"):
        super().__init__()
        self.release = release
        self.xyz = torch.zeros(n, 3)
        self.covariance = torch.eye(3).repeat(n, 1, 1)
        self.o_raw = torch.nn.Parameter(torch.zeros(n), requires_grad="o_raw" in RELEASE_NAMES[release])
        self.sh_coeffs = torch.nn.Parameter(torch.zeros(n, 9, 3), requires_grad="sh_coeffs" in RELEASE_NAMES[release])
        self.v_raw = torch.nn.Parameter(torch.zeros(n, 3), requires_grad="v_raw" in RELEASE_NAMES[release])

    def named_release_parameters(self):
        for name in RELEASE_NAMES[self.release]:
            yield name, getattr(self, name)
