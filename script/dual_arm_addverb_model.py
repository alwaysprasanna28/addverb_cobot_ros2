"""Addverb dual-arm kinematic and inertial model definitions.

This module encodes the 6-DOF chain parameters from
addverb_cobot_description/urdf/heal.urdf.xacro and wraps them for
left/right dual-arm usage.

Full inertia tensors (including off-diagonal terms) and joint limits
are taken directly from the URDF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class LinkSpec:
    joint_name: str
    joint_origin_xyz: np.ndarray
    joint_axis: np.ndarray
    com_xyz: np.ndarray          # CoM relative to link frame (from URDF <origin> in <inertial>)
    mass: float
    inertia_matrix: np.ndarray   # Full 3x3 inertia tensor (from URDF ixx/ixy/ixz/iyy/iyz/izz)
    q_min: float = -np.pi       # Joint lower limit (from URDF <limit>)
    q_max: float = np.pi        # Joint upper limit (from URDF <limit>)


@dataclass
class RobotModel:
    links: List[LinkSpec]
    base_xyz: np.ndarray
    base_rot: np.ndarray         # 3x3 rotation matrix for base orientation
    ee_offset_xyz: np.ndarray


@dataclass
class DualArmModel:
    left: RobotModel
    right: RobotModel


def _base_link_specs() -> List[LinkSpec]:
    """Parameters copied from heal.urdf.xacro (joint origins/axes + full link inertials + limits)."""
    return [
        LinkSpec(
            joint_name="joint1",
            joint_origin_xyz=np.array([0.0, 0.0, 0.0]),
            joint_axis=np.array([0.0, 0.0, 1.0]),
            com_xyz=np.array([-0.012435, -2.5345e-05, 0.28172]),
            mass=5.0617,
            inertia_matrix=np.array([
                [0.011934, -4.2772e-06, 0.0016521],
                [-4.2772e-06, 0.011427, 3.2849e-06],
                [0.0016521, 3.2849e-06, 0.0091785],
            ]),
            q_min=-np.pi,
            q_max=np.pi,
        ),
        LinkSpec(
            joint_name="joint2",
            joint_origin_xyz=np.array([0.0, 0.0, 0.346]),
            joint_axis=np.array([-1.0, 0.0, 0.0]),
            com_xyz=np.array([-0.10679, 3.7919e-06, 0.15236]),
            mass=1.2353,
            inertia_matrix=np.array([
                [0.0057954, 3.2114e-08, 3.5658e-06],
                [3.2114e-08, 0.0048706, -6.4438e-07],
                [3.5658e-06, -6.4438e-07, 0.0016259],
            ]),
            q_min=-0.75,
            q_max=2.05,
        ),
        LinkSpec(
            joint_name="joint3",
            joint_origin_xyz=np.array([0.0, 0.0, 0.305]),
            joint_axis=np.array([1.0, 0.0, 0.0]),
            com_xyz=np.array([-0.017248, 0.035025, -2.2087e-05]),
            mass=3.5577,
            inertia_matrix=np.array([
                [0.0066247, -0.00095453, 1.793e-06],
                [-0.00095453, 0.0050017, 1.7882e-06],
                [1.793e-06, 1.7882e-06, 0.0060636],
            ]),
            q_min=-1.0,
            q_max=1.0,
        ),
        LinkSpec(
            joint_name="joint4",
            joint_origin_xyz=np.array([-0.0009127, 0.36813, 0.0]),
            joint_axis=np.array([0.0, 1.0, 0.0]),
            com_xyz=np.array([-0.047999, -0.10939, 4.4809e-05]),
            mass=1.31,
            inertia_matrix=np.array([
                [0.0014749, 0.00050231, 1.7564e-06],
                [0.00050231, 0.0016894, -1.9993e-06],
                [1.7564e-06, -1.9993e-06, 0.0017778],
            ]),
            q_min=-np.pi,
            q_max=np.pi,
        ),
        LinkSpec(
            joint_name="joint5",
            joint_origin_xyz=np.array([0.0, 0.0, 0.0]),
            joint_axis=np.array([0.5, 0.86603, 0.0]),
            com_xyz=np.array([-0.0015193, -0.0027725, -0.044032]),
            mass=1.1965,
            inertia_matrix=np.array([
                [0.0010777, -3.7989e-05, 2.9305e-05],
                [-3.7989e-05, 0.0010311, 4.9507e-05],
                [2.9305e-05, 4.9507e-05, 0.0013412],
            ]),
            q_min=-np.pi,
            q_max=np.pi,
        ),
        LinkSpec(
            joint_name="joint6",
            joint_origin_xyz=np.array([0.0, 0.0, -0.0975]),
            joint_axis=np.array([0.0, 0.0, -1.0]),
            # URDF has no <inertial> for end-effector link.
            # Use small values for numerical stability in RNEA.
            com_xyz=np.array([0.0, 0.0, -0.01]),
            mass=0.1,
            inertia_matrix=np.diag([1e-4, 1e-4, 1e-4]),
            q_min=-np.pi,
            q_max=np.pi,
        ),
    ]


def _rpy_to_rotation(rpy: np.ndarray) -> np.ndarray:
    """Convert roll-pitch-yaw (XYZ extrinsic) to 3x3 rotation matrix."""
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def build_dual_arm_model(
    left_base_xyz: np.ndarray,
    right_base_xyz: np.ndarray,
    ee_offset_xyz: np.ndarray,
    left_base_rpy: np.ndarray = None,
    right_base_rpy: np.ndarray = None,
) -> DualArmModel:
    left_links = _base_link_specs()
    right_links = _base_link_specs()

    left_rot = _rpy_to_rotation(left_base_rpy) if left_base_rpy is not None else np.eye(3)
    right_rot = _rpy_to_rotation(right_base_rpy) if right_base_rpy is not None else np.eye(3)

    return DualArmModel(
        left=RobotModel(
            links=left_links,
            base_xyz=np.array(left_base_xyz, dtype=float),
            base_rot=left_rot,
            ee_offset_xyz=np.array(ee_offset_xyz, dtype=float),
        ),
        right=RobotModel(
            links=right_links,
            base_xyz=np.array(right_base_xyz, dtype=float),
            base_rot=right_rot,
            ee_offset_xyz=np.array(ee_offset_xyz, dtype=float),
        ),
    )


def approximate_joint_inertia(robot: RobotModel) -> np.ndarray:
    """Coarse positive-definite joint-space diagonal inertia approximation.

    Kept for backward compatibility / quick sanity checks. Not used by the
    proper RNEA-based dynamics.
    """
    diag = []
    running_mass = 0.0
    for link in reversed(robot.links):
        running_mass += max(link.mass, 0.05)
        diag.append(0.04 * running_mass + float(np.trace(link.inertia_matrix)))
    return np.array(list(reversed(diag)))
