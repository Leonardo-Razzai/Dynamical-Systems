# Dynamical Systems Simulator

A Python package for numerical simulation and visualization of 2D dynamical systems using the 4th-order Runge-Kutta method.

## Features

- **Numerical Integration**: 4th-order Runge-Kutta method for accurate ODE solutions
- **Multiple Visualization Tools**:
  - Vector field plots
  - Phase portraits with trajectories
  - Time evolution plots (x vs t)
- **Flexible Initial Conditions**: Support for multiple initial conditions simultaneously
- **Customizable Plots**: Adjustable figure sizes and grid resolutions

## Installation

### Prerequisites

- Python 3.6 or higher
- Required packages: `numpy`, `matplotlib`

### Quick Install

```bash
pip install numpy matplotlib
```

### Using the Package

1. Download the Python file (`dynamical_systems.py`) to your project directory
2. Import the module in your Python script:

```python
from dynamical_systems import Dynamical_system, odeRK4
```

## Quick Start

### Example 1: Simple Harmonic Oscillator

```python
import numpy as np
import matplotlib.pyplot as plt
from dynamical_systems import Dynamical_system

# Define the harmonic oscillator: dx/dt = y, dy/dt = -x
def harmonic_oscillator(state, t):
    x, y = state
    return np.array([y, -x])

# Set initial conditions: multiple starting points
init_cond = np.array([[1, 0, -1],  # x0 values
                      [0, 1, 0]])   # y0 values

# Create and simulate the system
system = Dynamical_system(harmonic_oscillator, init_cond, tmax=10, N=1000)
system.compute_sol()

# Visualize results
system.plot_phase_portrait()
system.plot_x_t()  # Plot x(t) for first initial condition
```

### Example 2: Damped Oscillator

```python
# Damped oscillator: dx/dt = y, dy/dt = -0.5*y - x
def damped_oscillator(state, t):
    x, y = state
    return np.array([y, -0.5*y - x])

# Single initial condition
init_cond = np.array([[2], [0]])  # x0=2, y0=0

system = Dynamical_system(damped_oscillator, init_cond, tmax=15, N=1500)
system.compute_sol()

system.plot_phase_portrait(Nx=25, Ny=25, figsize=(10, 8))
system.plot_x_t(figsize=(10, 6))
```

### Example 3: Nonlinear System (Van der Pol Oscillator)

```python
# Van der Pol oscillator: dx/dt = y, dy/dt = μ(1-x²)y - x
def van_der_pol(state, t, mu=1.0):
    x, y = state
    return np.array([y, mu*(1-x**2)*y - x])

# Multiple initial conditions around the origin
theta = np.linspace(0, 2*np.pi, 8)
init_cond = np.array([2*np.cos(theta), 2*np.sin(theta)])

system = Dynamical_system(lambda state, t: van_der_pol(state, t, mu=2.0), 
                         init_cond, tmax=20, N=2000)
system.compute_sol()

system.plot_phase_portrait(Nx=30, Ny=30)
```

## API Reference

### `Dynamical_system` Class

#### Constructor
```python
Dynamical_system(fun, init_cond, tmax, N)
```
- `fun`: Function that defines the derivatives `dx/dt = f(x, t)`
- `init_cond`: 2×n array of initial conditions `[[x01, x02, ...], [y01, y02, ...]]`
- `tmax`: Final simulation time
- `N`: Number of time steps

#### Methods
- `compute_sol()`: Run the numerical simulation
- `plot_field(Nx, Ny, figsize)`: Plot only the vector field
- `plot_x_t(figsize, init_cond_index)`: Plot x(t) for a specific initial condition
- `plot_phase_portrait(Nx, Ny, figsize)`: Plot vector field with trajectories

### `odeRK4` Function
```python
odeRK4(fun, x, t, dt)
```
4th-order Runge-Kutta integration step.

## Defining Your Own Systems

Create a function that returns the derivatives for your 2D system:

```python
def my_system(state, t, *parameters):
    x, y = state
    # Define your equations here
    dxdt = ...  # function of x, y, t, parameters
    dydt = ...  # function of x, y, t, parameters
    return np.array([dxdt, dydt])
```

The function should:
- Take a state vector `[x, y]` and time `t` as input
- Return a numpy array `[dx/dt, dy/dt]`
- Can include additional parameters using lambda functions

## Tips for Effective Use

1. **Time Steps**: Use more steps (`N`) for stiff systems or longer simulation times
2. **Initial Conditions**: Spread initial conditions to explore different regions of phase space
3. **Vector Field Resolution**: Increase `Nx`, `Ny` for smoother vector fields
4. **Figure Sizes**: Adjust `figsize` for better visualization of your specific system

## Common Dynamical Systems

The package can simulate various 2D systems:
- Linear systems (harmonic oscillators, damped systems)
- Nonlinear oscillators (Van der Pol, Duffing)
- Predator-prey models (Lotka-Volterra)
- Chemical reaction systems
- Electrical circuit models

## License

This is free to use for educational and research purposes. Please attribute the original authors if used in publications.

## Support

For issues or questions, please check that:
- Your derivative function returns a numpy array of shape (2,)
- Initial conditions are provided as a 2×n array
- All required packages are installed
- Time steps are sufficiently small for your system's dynamics