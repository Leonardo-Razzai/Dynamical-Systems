import numpy as np
import matplotlib.pyplot as plt

def odeRK4(fun, x: np.array, t: float, dt: float) -> np.array:
  f1 = fun(x, t)
  f2 = fun(x + f1*dt/2, t + dt/2)
  f3 = fun(x + f2*dt/2, t + dt/2)
  f4 = fun(x + f3*dt, t + dt)
  return x + dt/6 * (f1 + 2*f2 + 2*f3 + f4)

class Dynamical_system:
  '''
  Class for 2D dynamical systems.\n
  Input:
  - fun: function defining derivatives for 2d dynamical system
  - init_cond: initial conditions to system of diff. equations
  - tmax: solution runs from time 0 to time tmax
  - N: number of temporal divisions -> dt = tmax/N
  '''
  def __init__(self, fun, init_cond, tmax, N) -> None:
    self.fun = fun
    self.init_cond = init_cond
    self.N = N
    self.time = np.linspace(0, tmax, N)
    self.dt = tmax/N
    self.sol = np.zeros(shape=(len(self.time)+1, self.init_cond.shape[0], self.init_cond.shape[1]))
  
  def compute_sol(self):
    '''
    Compute solution by RK method to order 4.\n
    Save it in self.sol, where sol = sol[i, j, k] and:
    - i -> time
    - j -> coordinate (x = 0, y = 1)
    - k -> initial conditions
    '''
    self.sol[0] = self.init_cond
    for i, t in enumerate(self.time):
      self.sol[i+1] = odeRK4(self.fun, self.sol[i], t=t, dt=self.dt)
      
  def plot_field(self, Nx, Ny, figsize=(8, 8)):
    
    minx = np.min(self.sol[:, 0, :])
    maxx = np.max(self.sol[:, 0, :])
    x = np.linspace(minx, maxx, num = Nx)

    miny = np.min(self.sol[:, 1, :])
    maxy = np.max(self.sol[:, 1, :])
    y = np.linspace(miny, maxy, num = Ny)

    grid = np.array(np.meshgrid(x, y)) # points where field is drawn
    vector_field = self.fun(grid, 0) # vector field of dynamical system
    
    _, ax = plt.subplots(1, 1, figsize=figsize)
    ax.quiver(grid[0], grid[1], vector_field[0], vector_field[1], alpha=0.5) # display vector field
    ax.grid()
    ax.set_title('DynSys vector field')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
  
  def plot_x_t(self, figsize = (12, 7), init_cond_index = 0):
    _, ax = plt.subplots(1, 1, figsize = figsize)
    x = self.sol[:-1, 0, init_cond_index]
    t = self.time
    ax.plot(t, x)
    ax.set_title('x(t) plot')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.grid()
  
  def plot_phase_portrait(self, Nx = 20, Ny = 20, figsize=None):
    '''
    Draw phase portrait for this dynamical system
    Nx, Ny = number of points, along x and y, where vector field is drawn 
    '''
    minx = np.min(self.sol[:, 0, :])
    maxx = np.max(self.sol[:, 0, :])
    dx = maxx - minx
    x = np.linspace(minx, maxx, num = Nx)

    miny = np.min(self.sol[:, 1, :])
    maxy = np.max(self.sol[:, 1, :])
    dy = maxy - miny
    y = np.linspace(miny, maxy, num = Ny)

    grid = np.array(np.meshgrid(x, y)) # points where field is drawn
    vector_field = self.fun(grid, 0) # vector field of dynamical system
    
    dtot = dx + dy
    
    if figsize != None:
      _, ax = plt.subplots(1, 1, figsize=figsize)
    else:
      _, ax = plt.subplots(1, 1, figsize=(18 * dx/dtot, 18 * dy/dtot))
      
    ax.quiver(grid[0], grid[1], vector_field[0], vector_field[1], alpha=0.5) # display vector field
    for j in range(len(self.init_cond[0])):
      ax.plot(self.sol[:, 0, j], self.sol[:, 1, j], lw=2)
    
    ax.set_title('Phase Portrait')
    ax.grid()
    plt.show()