"""
Numerical simulation and visualization of 2D dynamical systems.

This module provides tools for solving and visualizing 2D dynamical systems
using the 4th-order Runge-Kutta method. It includes functionality for plotting
vector fields, phase portraits, and time evolution of the system.

Classes:
    Dynamical_system: Main class for simulating and visualizing 2D dynamical systems.

Functions:
    odeRK4: 4th-order Runge-Kutta method for numerical integration of ODEs.
"""

import numpy as np
import matplotlib.pyplot as plt

def odeRK4(fun, x: np.array, t: float, dt: float) -> np.array:
    """
    Perform one step of 4th-order Runge-Kutta numerical integration.
    
    Parameters:
    -----------
    fun : callable
        Function defining the derivatives: fun(x, t) -> dx/dt
    x : np.array
        Current state vector
    t : float
        Current time
    dt : float
        Time step size
        
    Returns:
    --------
    np.array
        State vector at next time step (t + dt)
    """
    f1 = fun(x, t)
    f2 = fun(x + f1*dt/2, t + dt/2)
    f3 = fun(x + f2*dt/2, t + dt/2)
    f4 = fun(x + f3*dt, t + dt)
    return x + dt/6 * (f1 + 2*f2 + 2*f3 + f4)


class Dynamical_system:
    """
    Class for simulating and visualizing 2D dynamical systems.
    
    This class provides methods to numerically solve 2D dynamical systems
    using the 4th-order Runge-Kutta method and visualize the results through
    vector fields, phase portraits, and time series plots.
    
    Parameters:
    -----------
    fun : callable
        Function defining the derivatives: fun(x, t) -> dx/dt
        Should take state vector x and time t, return derivatives
    init_cond : np.array
        Array of initial conditions with shape (2, n) where:
        - init_cond[0] contains initial x values
        - init_cond[1] contains initial y values  
        - n is the number of different initial conditions
    tmax : float
        Final time for simulation (runs from 0 to tmax)
    N : int
        Number of time steps (dt = tmax/N)
    
    Attributes:
    -----------
    fun : callable
        Function defining the dynamical system
    init_cond : np.array
        Initial conditions array
    N : int
        Number of time steps
    time : np.array
        Time array from 0 to tmax with N points
    dt : float
        Time step size
    sol : np.array
        Solution array with shape (N+1, 2, n_conditions)
        - First index: time step (including initial condition at index 0)
        - Second index: coordinate (0=x, 1=y)
        - Third index: initial condition index
    """
    
    def __init__(self, fun, init_cond, tmax, N) -> None:
        self.fun = fun
        self.init_cond = init_cond
        self.N = N
        self.time = np.linspace(0, tmax, N)
        self.dt = tmax/N
        self.sol = np.zeros(shape=(len(self.time)+1, self.init_cond.shape[0], 
                                 self.init_cond.shape[1]))
  
    def compute_sol(self):
        """
        Compute numerical solution using 4th-order Runge-Kutta method.
        
        The solution is stored in self.sol, where:
        sol[i, j, k] represents:
        - i: time index (0 = initial condition)
        - j: coordinate (0 = x, 1 = y)
        - k: initial condition index
        """
        self.sol[0] = self.init_cond
        for i, t in enumerate(self.time):
            self.sol[i+1] = odeRK4(self.fun, self.sol[i], t=t, dt=self.dt)
      
    def plot_field(self, Nx, Ny, figsize=(8, 8)):
        """
        Plot the vector field of the dynamical system.
        
        Parameters:
        -----------
        Nx : int
            Number of grid points in x-direction
        Ny : int
            Number of grid points in y-direction
        figsize : tuple, optional
            Figure size (width, height) in inches, default=(8, 8)
        """
        minx = np.min(self.sol[:, 0, :])
        maxx = np.max(self.sol[:, 0, :])
        x = np.linspace(minx, maxx, num=Nx)

        miny = np.min(self.sol[:, 1, :])
        maxy = np.max(self.sol[:, 1, :])
        y = np.linspace(miny, maxy, num=Ny)

        grid = np.array(np.meshgrid(x, y))  # points where field is drawn
        vector_field = self.fun(grid, 0)    # vector field of dynamical system
    
        _, ax = plt.subplots(1, 1, figsize=figsize)
        ax.quiver(grid[0], grid[1], vector_field[0], vector_field[1], alpha=0.5)
        ax.grid()
        ax.set_title('DynSys vector field')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()
  
    def plot_x_t(self, figsize=(12, 7), init_cond_index=0):
        """
        Plot the time evolution of the x-coordinate.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height) in inches, default=(12, 7)
        init_cond_index : int, optional
            Index of initial condition to plot, default=0
        """
        _, ax = plt.subplots(1, 1, figsize=figsize)
        x = self.sol[:-1, 0, init_cond_index]
        t = self.time
        ax.plot(t, x)
        ax.set_title('x(t) plot')
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.grid()
        plt.show()
  
    def plot_phase_portrait(self, Nx=20, Ny=20, figsize=None):
        """
        Plot phase portrait with vector field and trajectories.
        
        The phase portrait shows both the vector field (background arrows)
        and the solution trajectories for all initial conditions.
        
        Parameters:
        -----------
        Nx : int, optional
            Number of grid points in x-direction for vector field, default=20
        Ny : int, optional
            Number of grid points in y-direction for vector field, default=20
        figsize : tuple, optional
            Figure size (width, height) in inches. If None, automatically
            scales to maintain proper aspect ratio.
        """
        minx = np.min(self.sol[:, 0, :])
        maxx = np.max(self.sol[:, 0, :])
        dx = maxx - minx
        x = np.linspace(minx, maxx, num=Nx)

        miny = np.min(self.sol[:, 1, :])
        maxy = np.max(self.sol[:, 1, :])
        dy = maxy - miny
        y = np.linspace(miny, maxy, num=Ny)

        grid = np.array(np.meshgrid(x, y))  # points where field is drawn
        vector_field = self.fun(grid, 0)    # vector field of dynamical system
    
        dtot = dx + dy
    
        if figsize is not None:
            _, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            _, ax = plt.subplots(1, 1, figsize=(18 * dx/dtot, 18 * dy/dtot))
      
        ax.quiver(grid[0], grid[1], vector_field[0], vector_field[1], alpha=0.5)
        for j in range(len(self.init_cond[0])):
            ax.plot(self.sol[:, 0, j], self.sol[:, 1, j], lw=2)
    
        ax.set_title('Phase Portrait')
        ax.grid()
        plt.show()