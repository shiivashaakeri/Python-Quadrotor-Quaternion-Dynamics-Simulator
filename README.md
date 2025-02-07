# Python-Quadrotor-Quaternion-Dynamics-Simulator  
*A minimal, quaternion-based quadrotor dynamics model for robotics, control systems, and aerospace applications.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## 🚀 Quick Start
```bash
git clone https://github.com/yourusername/Python-Quadrotor-Quaternion-Dynamics-Simulator.git
pip install -r requirements.txt

```
import numpy as np
from rocket_dynamics import RocketDynamics

# Initialize with default parameters
params = {
    "m": 1.5,          
    "g": 9.81,         
    "J_b": np.diag([0.03, 0.03, 0.06]),
    "dt_ss": 0.1       
}

rocket = RocketDynamics(params)
initial_state = np.zeros(13)  # [position, velocity, quaternion, angular rate]
controls = np.array([0, 0, 15, 0, 0, 0])  # [F_x, F_y, F_z, τ_x, τ_y, τ_z]

# Simulate nonlinear dynamics
states = rocket.simulate_nonlinear(initial_state, controls, dt=0.1)```

