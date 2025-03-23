import numpy as np
# from scipy.fft import fft, fftfreq, fft2, ifft
import torch
from torch.fft import fft, fftfreq, fft2, ifft
torch.set_default_dtype(torch.float64)


def fourier_interp(fx, points, period = 1.0):
    """Fourier interpolation on [0, period) interval.

    Args:
        fx (array): uniformly spaced evaluations of f(x) on [0, period). Make sure
            x was spaced using linspace(0, period, n, endpoint=False).
        points (array): points at which to evaluate the interpolant.
        period (float): period of the function.

    Returns:
        array: len(points) array of interpolated values of f.
    """
    coeffs = fft(fx)
    size = len(coeffs)
    kn = fftfreq(size, period / size)
    eikx = torch.exp(2.j * torch.pi * torch.outer(points, kn))
    return torch.real(torch.matmul(eikx, coeffs) / size)

def fourier_interp_2d(fxy, points, x_period=1.0, y_period=1.0):
    """Fourier interpolation for 2D periodic functions.
    
    Args:
        fxy (array): 2D array of uniformly spaced evaluations of f(x,y) on the respective
            periods. Evaluations should be spaced using 
                X, Y = torch.meshgrid(x, y, indexing='xy')
                fxy = f(X, Y)
        points (array): Array of shape (N, 2) points (x,y) at which to evaluate the interpolant.
        x_period (float): period in x-direction.
        y_period (float): period in y-direction.
        
    Returns:
        array: (N,) array of interpolated values at given points.
    """
    # Get dimensions
    nx, ny = fxy.shape
    size = nx * ny
    
    # Compute 2D FFT
    coeffs = fft2(fxy)
    
    # Create frequency grids
    kx = fftfreq(nx, x_period / nx)
    ky = fftfreq(ny, y_period / ny)
    
    # Initialize result array
    result = torch.zeros(len(points))
    
    # Compute interpolation
    for ii, xx in enumerate(points):
        eikx = torch.exp(2.0j * torch.pi * (kx * xx[0]))
        eiky = torch.exp(2.0j * torch.pi * (ky * xx[1]))
        res = torch.matmul(torch.matmul(coeffs, eikx), eiky)
        result[ii] = (res / size).real
    
    return (result).real

def fourier_differentiation(fx, period=2*torch.pi):
    """ Differentiate a periodic function using Fourier (spectral) differentiation.

    Args:
        fx (array): (N,) array of evaluations of a periodic function at uniformly
            spaced points on the period. Evaluations should be spaced using 
                linspace(0, period, N, endpoint=False).
        period (float, optional): period of f. Defaults to 2*torch.pi.

    Returns:
        array: (N,) array of derivative evaluations at the interpolation points.
    """
    coeffs = fft(fx)
    N = len(fx)
    fft_freqs = fftfreq(N,  d=period/N)
    df_by_dx = ifft(2* 1j * torch.pi * coeffs * fft_freqs)
    return df_by_dx.real
