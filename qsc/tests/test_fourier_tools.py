import matplotlib.pyplot as plt
import torch
import numpy as np
from qsc.fourier_tools import (fourier_interp1d, fourier_interp2d,
                               fourier_differentiation, fourier_interp1d_regular_grid)
    
def test_fourier_interp():
    """ 1D fourier interpolation """

    # Example usage
    n = 30
    period = 2*torch.pi*5
    x = torch.tensor(np.linspace(0, period, n, endpoint=False))
    fx = torch.sin(4*x/5) + torch.cos(x/5)  # periodic function
    y = torch.linspace(0, period, 500)
    fy = fourier_interp1d(fx, y, period=period)

    # interpolation error
    err = torch.max(torch.abs(fx - fourier_interp1d(fx, x, period=period)))
    print('1D interpolation error', err)
    assert err < 1e-14, "1D interpolation failed"

    plt.plot(x, fx.detach().numpy(), 'o', label='Sample Points')
    plt.plot(y, fy.detach().numpy(), '-', label='Interpolated')
    plt.legend()
    plt.show()

def test_fourier_interp1d_regular_grid():
    """ 1D fourier interpolation """

    # Example usage
    n = 30
    period = 2*torch.pi*5
    x = torch.tensor(np.linspace(0, period, n, endpoint=False))
    fx = torch.sin(4*x/5) + torch.cos(x/5)  # periodic function
    m = 500
    y = torch.linspace(0, period, m)
    fy = fourier_interp1d_regular_grid(fx, m)

    # interpolation error
    err = torch.max(torch.abs(fx - fourier_interp1d_regular_grid(fx, n)))
    print('1D interpolation error', err)
    assert err < 1e-14, "1D interpolation failed"

    plt.plot(x, fx.detach().numpy(), 'o', label='Sample Points')
    plt.plot(y, fy.detach().numpy(), '-', label='Interpolated')
    plt.legend()
    plt.show()

    # test a multi-output function
    n = 30
    period = 2*torch.pi*5
    x = torch.tensor(np.linspace(0, period, n, endpoint=False))
    fx1 = torch.sin(4*x/5) + torch.cos(x/5)  # periodic function
    fx2 = torch.sin(2*x/5) + 0.1*torch.cos(x/5)  # periodic function
    fx = torch.stack((fx1, fx2)).T

    m = 500
    y = torch.linspace(0, period, m)
    fy = fourier_interp1d_regular_grid(fx, m)

    # interpolation error
    err = torch.max(torch.abs(fx - fourier_interp1d_regular_grid(fx, n)))
    print('1D interpolation error', err)
    assert err < 1e-14, "1D interpolation failed"

    plt.plot(x, fx.detach().numpy(), 'o', label='Sample Points')
    plt.plot(y, fy.detach().numpy(), '-', label='Interpolated')
    plt.legend()
    plt.show()

def test_fourier_interp2d():

    """ 2D fourier interpolation """

    # Parameters
    n = 40  # Number of sample points in each dimension
    period = 2*torch.pi*2  # Period in both directions

    # Create sample grid
    x = torch.tensor(np.linspace(0, period, n, endpoint=False))
    y = torch.tensor(np.linspace(0, period, n, endpoint=False))
    X, Y = torch.meshgrid(x, y, indexing='xy')

    # Create a 2D periodic function
    def f(x, y):
        return torch.sin(x/2) * torch.cos(y/2) + 0.5 * torch.sin(2*x/2 + y/2)

    # Sample the function on the grid
    fxy = f(X, Y)

    # Create a finer grid for interpolation
    n_fine = 100
    x_fine = torch.tensor(np.linspace(0, period, n_fine, endpoint=False))
    y_fine = torch.tensor(np.linspace(0, period, n_fine, endpoint=False))
    X_fine, Y_fine = torch.meshgrid(x_fine, y_fine, indexing='xy')

    # Prepare points for interpolation
    points = torch.column_stack((X_fine.flatten(), Y_fine.flatten()))

    # Interpolate
    fxy_interp = fourier_interp2d(fxy, points, x_period=period, y_period=period)
    fxy_interp = fxy_interp.reshape(X_fine.shape)

    # Calculate true values for comparison
    fxy_true = f(X_fine, Y_fine)

    # Calculate interpolation error
    interp_error = torch.max(torch.abs(fxy_true - fxy_interp))
    print(f'2D interpolation error: {interp_error}')
    assert interp_error <= 1e-14, "2D interpolation failed"

    # Plot the results
    fig = plt.figure(figsize=(15, 5))

    # Sample points
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X.detach().numpy(), Y.detach().numpy(), fxy.detach().numpy(), cmap='viridis', alpha=0.7)
    ax1.set_title('Original Sample Points')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('f(x,y)')

    # Interpolated surface
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X_fine.detach().numpy(), Y_fine.detach().numpy(), fxy_interp.detach().numpy(), cmap='viridis', alpha=0.7)
    ax2.set_title('Interpolated Surface')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('f(x,y)')

    # Error plot
    ax3 = fig.add_subplot(133, projection='3d')
    error = torch.abs(fxy_true - fxy_interp)
    ax3.plot_surface(X_fine.detach().numpy(), Y_fine.detach().numpy(), error.detach().numpy(), cmap='hot', alpha=0.7)
    ax3.set_title(f'Interpolation Error (Max: {interp_error:.6f})')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Error')

    plt.tight_layout()
    plt.show()


def test_fourier_differentiation():
    """ Fourier differentiation """

    def func(phi):
        return 3.2*torch.cos(2*torch.pi*3*phi/period) + 0.7*torch.sin(2*torch.pi*7*phi/period)
    def dfunc(phi):
        return -2*torch.pi*3*3.2*torch.sin(2*torch.pi*3*phi/period)/period + 2*torch.pi*7*0.7*torch.cos(2*torch.pi*7*phi/period)/period
    
    # differentiate with the fft
    period = 0.5
    x = torch.tensor(np.linspace(0, period, 32, endpoint=False))
    fx = func(x)
    df_by_dx = dfunc(x)
    df_by_dx_fourier = fourier_differentiation(fx, period)

    err = torch.max(torch.abs(df_by_dx - df_by_dx_fourier))
    print('differentiation error:', err)
    assert err <=1e-12, "Differentation failed"

    plt.plot(x.detach().numpy(), df_by_dx.detach().numpy(), label='exact')
    plt.plot(x.detach().numpy(), df_by_dx_fourier.detach().numpy(), label='fourier')
    plt.title("Fourier Differentiation")
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    test_fourier_interp()
    test_fourier_interp2d()
    test_fourier_interp1d_regular_grid()
    test_fourier_differentiation()