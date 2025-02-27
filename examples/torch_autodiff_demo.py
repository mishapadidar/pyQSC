import torch
from torch.autograd.functional import jacobian
from functools import partial
torch.set_default_dtype(torch.float64)

class QSCSimple(torch.nn.Module):
    """A simple class that vaguely mimics the structure of PyQSC to demonstrate the development
    and usage of autodiff through torch.

    The class has degrees of freedom, x, and parameters, sigma, defined implicitly by
    a solving an equation,
        r(sigma, x) = 0.
    The class has loss functions that depend sigma and x, or just x.
    
    We do not use pure autodiff to differentiate sigma with respec to x. Instead, we opt for a hybrid
    approach where we use autodiff to build the adjoint system, and then solve the adjoint system to get
    dsigma/dx. This is more stable than autodiffing through the solve. However, this means that we have
    to explicitly use the chain rule when differentiating loss fucntions with respect to x. For example,
    suppose we have a loss function J(sigma(x), x). To compute the derivative,
        dJ/dx = dJ/dsigma * dsigma/dx + dJ/dx,
    we use autodiff to get dJ/dsigma and dJ/dx, and we solve the adjoint system to compute dsigma/dx.
    
    """
    def __init__(self, x):
        super().__init__()
        self.set_dofs(x)

    def calculate(self):
        self.kappa = self.compute_kappa()
        self.solve_system()
    
    def compute_kappa(self):
        # intermediate quantity
        return self.x[:2]**2
    
    def _residual(self, sigma):
        """ residual function r(sigma, x) = 0 """
        return self.kappa - 3.1*sigma**3
    
    def _jacobian(self, sigma):
        """ jacobian of residual function wrt sigma """
        # jacobian wrt sigma
        return - 9.3 * torch.diag(sigma**2)
    
    def solve_system(self):
        """ Compute sigma(x) by solving r(sigma, x) = 0"""
        # save values not gradients!
        sol = (self.kappa / 3.1)**(1/3)
        self.sigma = torch.nn.Parameter(torch.clone(sol).detach())

    def dsigma_by_dx_jvp(self, v):
        """ Adjoint system: compute the jacobian vector product transpose(dsigma/dx) * v """
        # solve (1) for lambda
        dr_by_dsigma = self._jacobian(self.sigma)
        _lambda = torch.linalg.solve(dr_by_dsigma.T, - v)

        # use autodiff to compute transpose(dr/dx) * v
        r = self._residual(self.sigma)
        dsigma = torch.autograd.grad(r, self.x, grad_outputs=_lambda, retain_graph=True)

        return dsigma
    
    def set_dofs(self, x):
        """ update the degrees of freedom, x."""
        self.x = torch.nn.Parameter(torch.clone(x).detach())
        self.calculate()

    def B_cartesian(self):
        B = torch.stack([torch.stack([self.x[0],self.x[1]]),
                         torch.stack([self.x[1],self.x[2]]),
                         torch.stack([self.x[0],self.x[1]]),
                         torch.stack([self.x[1],self.x[2]])]) # (n, 2)
        return B
    
    def B_loss(self, B_target):
        """ A loss function that depends only on x"""
        B = self.B_cartesian()
        loss = torch.sum((B - B_target)**2)
        return loss
    
    def dB_loss_by_dx(self, B_target):
        """ Gradient of B_loss wrt x"""
        loss = self.B_loss(B_target)
        loss.backward()
        return self.x.grad
    
    def C_loss(self):
        """ A loss function that depends on sigma and x. """
        return torch.sum(self.sigma**2) + torch.sum(self.x**2)
    
    def dC_loss_by_dx(self):
        """ Gradient of C_loss wrt x"""
        loss = self.C_loss()
        loss.backward()
        # chain rule
        partialC_by_partialx = self.x.grad # (n_x,)
        partialC_by_partialsigma = self.sigma.grad # (n_sigma,)
        partialC_by_partialx_indirect = self.dsigma_by_dx_jvp(partialC_by_partialsigma)[0]
        dC_by_dx = partialC_by_partialx_indirect + partialC_by_partialx
        return dC_by_dx
    
def finite_difference(f, x, eps=1e-6, *args, **kwargs):
    """Approximate jacobian with central difference.
    You should always clone and detach x before passing it to this function to prevent
    conflicts with aliasing and torch's autodiff graph.
        finite_difference(f, torch.clone(x).detach())

    Args:
        f (function): function to differentiate, can be scalar valued or 1d-array
            valued.
        x (tensor): input to f(x) at which to take the gradient.
        eps (float, optional): finite difference step size. Defaults to 1e-6.

    Returns:
        _type_: _description_
    """
    x = torch.clone(x)
    jac_est = []
    for i in range(len(x)):
        x[i] += eps
        fx = f(x, *args, **kwargs)
        x[i] -= 2*eps
        fy = f(x, *args, **kwargs)
        x[i] += eps
        jac_est.append((fx-fy)/(2*eps))
    return torch.stack(jac_est)


if __name__ == "__main__":
    x = torch.tensor([1.2, 2.34, 3.7])
    model = QSCSimple(x)

    print("Model Parameters")
    for param, val in model.named_parameters():
        print(param, val)
    print("")

    # compute dsigma/dx * z
    z = torch.tensor([1.124124, 2.421])
    jvp = model.dsigma_by_dx_jvp(z)[0].detach()

    # compare to finite difference
    def fd_obj(x):
        model.set_dofs(x)
        return model.sigma.detach()
    dsigma_by_dx_fd = finite_difference(fd_obj, torch.clone(model.x.detach()), eps=1e-5)
    jvp_fd = torch.matmul(dsigma_by_dx_fd, z)
    err = torch.max(torch.abs(jvp - jvp_fd))
    print("dsigma/dx jvp err ", err.item())

    # a loss function that just depends on x
    B_target = torch.ones((4,2))
    print("B_loss", model.B_loss(B_target).item())

    # derivative of the loss
    model.zero_grad() # zero-out the gradients
    dB_loss_by_dx = model.dB_loss_by_dx(B_target)
    def fd_obj(x):
        model.set_dofs(x)
        return model.B_loss(B_target).detach()
    dB_loss_by_dx_fd = finite_difference(fd_obj, torch.clone(model.x.detach()), eps=1e-3)
    err = torch.max(torch.abs(dB_loss_by_dx - dB_loss_by_dx_fd))
    print("B_loss gradient err", err.item())

    # a loss function that depends on x and sigma
    print("C_loss", model.C_loss().item())

    # derivative of the loss
    model.zero_grad() # zero-out the gradients
    dC_loss_by_dx = model.dC_loss_by_dx()
    def fd_obj(x):
        model.set_dofs(x)
        return model.C_loss().detach()
    dC_loss_by_dx_fd = finite_difference(fd_obj, torch.clone(model.x.detach()), eps=1e-5)
    err = torch.max(torch.abs(dC_loss_by_dx - dC_loss_by_dx_fd))
    print("C_loss gradient err", err.item())