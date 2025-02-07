# Python-Quadrotor-Quaternion-Dynamics-Simulator  
*A minimal, quaternion-based quadrotor dynamics model for robotics, control systems, and aerospace applications.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## Installation
```bash
git clone https://github.com/shiivashaakeri/Python-Quadrotor-Quaternion-Dynamics-Simulator.git
cd Python-Quadrotor-Quaternion-Dynamics-Simulator
pip install -r requirements.txt
```

## Basic Usage

```py
import numpy as np
from rocket_dynamics import RocketDynamics

# Initialize system
params = {
    "m": 1.5,
    "g": np.array([0, 0, -9.81]),
    "J_b": np.diag([0.03, 0.03, 0.06]),
    "dt_ss": 0.1
}

# Create dynamics model
rocket = RocketDynamics(params)

# Initial state: [position, velocity, quaternion, angular velocity]
initial_state = np.zeros(13)
initial_state[6:10] = np.array([1, 0, 0, 0])  # Identity quaternion

# Control inputs: [F_x, F_y, F_z, τ_x, τ_y, τ_z]
controls = np.array([0, 0, 14.715, 0, 0, 0])  # Hover command

# Simulate for 1 second
states = rocket.simulate_nonlinear(initial_state, controls, dt=0.1)
```


