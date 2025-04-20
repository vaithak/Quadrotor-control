## Implementation of control, planning, vision and state estimation algorithms for CrazyFile 2.0 quadrotor
### Subsystems implemented
- Stereo Visual Inertial Odometry using Error State Kalman Filter
- [Geometric Tracking Control of a Quadrotor UAV on SE(3)](https://mathweb.ucsd.edu/~mleok/pdf/LeLeMc2010_quadrotor.pdf)
- Dynamic A* path planning with minimum acceleration (cubic spline) fitting for smooth trajectories.

### Results
- On Crazyflie 2.0 drone

https://github.com/user-attachments/assets/945503b9-95c1-4f85-b742-c89b5855b3f3

- On simulation

[Screencast from 04-20-2025 04:19:41 PM.webm](https://github.com/user-attachments/assets/05c1b60a-dbf6-45c4-b9d4-08a99037a25d)  


### Usage:
```sh
cd quadrotor_control
pip install -e .
python quadrotor_control/code/sandbox.py
```

**This was done as a course project for MEAM 6200: Advanced Robotics, at UPenn, and the demo simulator is provided by the teaching staff.**
