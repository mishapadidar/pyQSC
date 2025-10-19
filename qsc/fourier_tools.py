import numpy as np
# from scipy.fft import fft, fftfreq, fft2, ifft
import torch
from torch.fft import fft, fftfreq, fft2, ifft, fftshift
torch.set_default_dtype(torch.float64)


def fourier_interp1d(fx, points, period = 1.0):
    """Fourier interpolation on [0, period) interval. For faster evaluation
    of the interpolant on a regular grid of evaluation points, use
    fourier_interp1d_regular_grid.

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

def fourier_coeffs(fx):
    """Compute the Fourier coefficients of a periodic function given
    uniformly spaced evaluations, i.e. ak and bk in
    f(x) = sum_{k=0}^{infty} a[k] cos(2 pi k x / period) + b[k] sin(2 pi k x / period)
    
    Args:
        fx (array): uniformly spaced evaluations of f(x) on [0, period). Make sure
        x was spaced using linspace(0, period, n, endpoint=False).

    Returns:
        ak (array): cosine coefficients
        bk (array): sine coefficients
    """
    fhat = fft(fx)
    size = len(fx)


    # 1 / n is the cofficient from the inverse FFT
    # 2 comes from taking only the positive frequencies
    ak = (2 / size) * fhat.real
    ak[0] = ak[0] / 2  # DC component should not be doubled
    bk = (-2 / size) * fhat.imag

    # only keep positive frequencies
    ak = fftshift(ak)[size//2:]
    bk = fftshift(bk)[size//2:]

    return ak, bk

def fourier_interp1d_regular_grid(fx, m):
    """
    Interpolate a periodic function onto a uniform grid. This function uses
    the fft and ifft to accelerate the interpolant to O(nlogn + mlogm) for 
    n interpolation knots and m evaluation points. f can be a multi-output
    function.
    
    Args:
        fx (array): (n,) or (n,d) array of evaluations of the function for a function
        with d outputs. Evaluations should be taken on a uniform grid over the period,
        exclusive of the right endpoint, i.e.
            x = linspace(a, b, n, endpoint=False)
            fx = f(x)
        m (int): Number of points to evaluate the interpolant at. Must be greater than
            or equal to n.
        
    Returns:
        interpolated_signal: (m,d) array of interpolated values.
    """
    ndim = fx.ndim
    if ndim == 1:
        fx = fx.reshape((-1,1))
    
    n, d = fx.shape
    pad = m - n
    if pad < 0:
        raise ValueError("m must be >= the number of interpolation points.")

    fx_interp = torch.zeros((m, d))
    for ii in range(d):
        # compute FFT
        coeffs = fft(fx[:,ii])
        
        # split and pad the spectrum
        half_len = n // 2
        padded_coeffs = torch.cat([coeffs[:half_len], torch.zeros(pad), coeffs[half_len:]])
        
        # scale factor
        scale_factor = m / n
        
        # ifft with scaling
        interpolated_signal = ifft(padded_coeffs) * scale_factor
        fx_interp[:,ii] = torch.real(interpolated_signal)

    if ndim == 1:
        fx_interp = fx_interp.flatten()

    return fx_interp

def fourier_interp2d(fxy, points, x_period=1.0, y_period=1.0):
    """Fourier interpolation for 2D periodic functions.
    
    Args:
        fxy (array): 2D array of uniformly spaced evaluations of f(x,y) on the respective
            periods. Evaluations should be spaced using 
                X, Y = torch.meshgrid(x, y, indexing='ij')
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
        res = torch.matmul(torch.matmul(coeffs, eiky), eikx)
        result[ii] = (res / size).real
    
    return (result).real

def fourier_interp2d_regular_grid(fxy, m_x, m_y):
    """
    Interpolate a 2D periodic function onto a uniform grid.
    
    Args:
        fxy (array): (n_x, n_y) or (n_x, n_y, d) array of function evaluations.
        Evaluations should be spaced using 
                X, Y = torch.meshgrid(x, y, indexing='ij')
                fxy = f(X, Y)
        m_x (int): Number of points in x dimension. Should be greater than or equal
            to n_x.
        m_y (int): Number of points in y dimension. Should be greater than or equal
            to n_y.
    
    Returns:
        interpolated_signal: (m_x, m_y, d) array of interpolated values
    """
    ndim = fxy.ndim
    if ndim == 2:
        fxy = fxy[:,:,None]

    N, M, d = fxy.shape

    if (m_x < N) or (m_y < M):
        raise ValueError(f"m_x, m_y must be >= shape(fxy). Interpolate at more points.")

    fxy_interp = torch.zeros((m_x, m_y, d))
    for ii in range(d):
        F = fft2(fxy[:,:,ii])

        G = torch.zeros((m_x, m_y), dtype=complex)
        G[:int(N/2),:int(M/2)] = F[:int(N/2),:int(M/2)]
        G[:int(N/2),-int(M/2):] = F[:int(N/2),-int(M/2):]
        G[-int(N/2):,:int(M/2)] = F[-int(N/2):,:int(M/2)]
        G[-int(N/2):,-int(M/2):] = F[-int(N/2):,-int(M/2):]

        # I am ignoring the normalization factor by using fft2 to actually
        # compute ifft2
        B = torch.real(torch.conj(fft2(torch.conj(G))))/(N*M)

        fxy_interp[:,:,ii] = B

    if ndim == 2:
        fxy_interp = fxy_interp.squeeze(-1)
    
    return fxy_interp

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
