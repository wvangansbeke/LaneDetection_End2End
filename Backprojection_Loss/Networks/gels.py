"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
from torch.autograd import Function

class GELS(Function):
    @staticmethod
    def forward(ctx, A, b):
        u = torch.cholesky(torch.matmul(A.transpose(-1, -2), A), upper=True)
        ret = torch.cholesky_solve(torch.matmul(A.transpose(-1, -2), b), u, upper=True)
        ctx.save_for_backward(u, ret, A, b)
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        chol, x, a, b = ctx.saved_tensors
        z = torch.cholesky_solve(grad_output, chol, upper=True)
        xzt = torch.matmul(x, z.transpose(-1,-2))
        zx_sym = xzt + xzt.transpose(-1, -2)
        grad_A = - torch.matmul(a, zx_sym) + torch.matmul(b, z.transpose(-1, -2))
        grad_b = torch.matmul(a, z)
        return grad_A, grad_b
