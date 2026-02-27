# Dual-Arm Geometric Reduced Closed-Loop Control

## What this script does
- Implements the pseudocode flow from `paper_code/Screenshot_2026-02-25_16-53-37.jpg`.
- Left arm (`/`) is commanded in `gravity_comp_effort_controller`.
- Right arm (`/ar2`) is commanded through `/ar2/cartesian_impedance_controller/follow_joint_trajectory`.

## Files
- `dual_arm_geometric_control.py`: ROS2 runtime and controller orchestration
- `dual_arm_addverb_model.py`: Addverb model parameters from URDF
- `dual_arm_math.py`: FK/Jacobian/reduced-dynamics/integrator/IK math
- `dual_arm_geometric_config.yaml`: tasks, gains, limits, runtime

## Prerequisites
Run in a shell where both ROS and workspace are sourced:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
source ~/code/addverb_cobot/cobot_ros2-main/cobot_ros2/install/setup.bash
```

If `addverb_cobot_msgs` import fails, build once:

```bash
colcon build --symlink-install --packages-select addverb_cobot_msgs
source install/setup.bash
```

## Run
Dry run (math and logging only):

```bash
python3 cobot_ros2/script/dual_arm_geometric_control.py --dry-run --task push
```

Live run:

```bash
python3 cobot_ros2/script/dual_arm_geometric_control.py --task push
```

Optional:

```bash
python3 cobot_ros2/script/dual_arm_geometric_control.py --task pull --rate 40
```

## Notes
- Script auto-switches controllers on both controller managers.
- Left controller receives zero effort commands while gravity compensation handles payload.
- Right controller receives short-horizon action goals each loop step.
- Logs are written to the configured CSV path.
