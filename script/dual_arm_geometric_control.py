#!/usr/bin/env python3
"""Dual-arm geometric reduced closed-loop controller for Addverb cobot.

Left arm (passive): gravity compensation via gravity_comp_effort_controller
Right arm (active): cartesian impedance via action goals

Implements Algorithm 1 from dual_arm_robot_control.pdf, adapted for
the Addverb Heal cobot ROS2 hardware interface.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

import rclpy
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectoryPoint

from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import SwitchController

from dual_arm_addverb_model import build_dual_arm_model
from dual_arm_math import (
    ClosedLoopState,
    GeometricIntegrator,
    ImpedanceController,
    ImpedanceGains,
    IntegratorLimits,
    ReducedDynamics,
    TrajectoryReference,
    forward_kinematics,
    quintic_trajectory,
)


JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


@dataclass
class RuntimeConfig:
    rate_hz: float
    output_csv: str
    warmup_timeout_sec: float
    sensor_blend_alpha: float


class DualArmGeometricNode(Node):
    def __init__(self, cfg: Dict, task_name: str, dry_run: bool):
        super().__init__("dual_arm_geometric_control")
        self.cfg = cfg
        self.task_name = task_name
        self.dry_run = dry_run

        self.left_joint_map: Dict[str, float] = {}
        self.right_joint_map: Dict[str, float] = {}

        self.sub_left = self.create_subscription(JointState, "/joint_states", self._left_cb, 10)
        self.sub_right = self.create_subscription(JointState, "/ar2/joint_states", self._right_cb, 10)

        self.left_effort_pub = self.create_publisher(Float64MultiArray, "/gravity_comp_effort_controller/commands", 10)

        self.right_action = ActionClient(
            self,
            FollowJointTrajectory,
            "/ar2/cartesian_impedance_controller/follow_joint_trajectory",
        )

        self.switch_left_client = self.create_client(SwitchController, "/controller_manager/switch_controller")
        self.switch_right_client = self.create_client(SwitchController, "/ar2/controller_manager/switch_controller")

    def _left_cb(self, msg: JointState):
        self.left_joint_map = {n: p for n, p in zip(msg.name, msg.position)}

    def _right_cb(self, msg: JointState):
        self.right_joint_map = {n: p for n, p in zip(msg.name, msg.position)}

    def get_q(self, mapping: Dict[str, float]) -> Optional[np.ndarray]:
        vals = []
        for j in JOINTS:
            if j not in mapping:
                return None
            vals.append(mapping[j])
        return np.array(vals, dtype=float)

    def wait_for_joint_states(self, timeout_sec: float) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.get_q(self.left_joint_map) is not None and self.get_q(self.right_joint_map) is not None:
                return True
        return False

    def _switch_once(self, client, activate: List[str], deactivate: List[str]) -> bool:
        if not client.wait_for_service(timeout_sec=5.0):
            return False
        req = SwitchController.Request()
        req.activate_controllers = activate
        req.deactivate_controllers = deactivate
        req.strictness = 2  # STRICT
        req.start_asap = True
        req.timeout.sec = 3
        fut = client.call_async(req)
        while rclpy.ok() and not fut.done():
            rclpy.spin_once(self, timeout_sec=0.05)
        if not fut.done() or fut.result() is None:
            return False
        return bool(fut.result().ok)

    def switch_required_controllers(self) -> bool:
        left_deactivate = [
            "velocity_controller",
            "effort_controller",
            "ptp_joint_controller",
            "ptp_tcp_controller",
            "joint_jogging_controller",
            "cartesian_jogging_controller",
            "joint_impedance_controller",
            "cartesian_impedance_controller",
            "free_drive_controller",
            "recorder_controller",
        ]
        right_deactivate = [
            "velocity_controller",
            "effort_controller",
            "gravity_comp_effort_controller",
            "ptp_joint_controller",
            "ptp_tcp_controller",
            "joint_jogging_controller",
            "cartesian_jogging_controller",
            "joint_impedance_controller",
            "free_drive_controller",
            "recorder_controller",
        ]

        ok_left = self._switch_once(self.switch_left_client, ["gravity_comp_effort_controller"], left_deactivate)
        ok_right = self._switch_once(self.switch_right_client, ["cartesian_impedance_controller"], right_deactivate)
        return ok_left and ok_right

    def publish_left_gravity_hold(self):
        """Publish zero additional effort to the gravity compensation controller.

        HARDWARE CONTEXT (from gravity_comp_effort_controller.cpp):
        This controller is a raw effort pass-through — it sends commanded_effort
        directly to command_interfaces. The "gravity compensation" naming refers
        to the firmware-level behavior of the Addverb cobot hardware, which
        internally compensates for gravity when receiving effort commands.

        Sending [0]*6 means: "apply only firmware-level gravity compensation,
        no additional user effort." This effectively holds the arm in place
        against gravity, which is appropriate for the paper's τ_II = 0
        assumption (the passive arm is unactuated at the algorithmic level,
        while the hardware prevents gravitational collapse).
        """
        msg = Float64MultiArray()
        msg.data = [0.0] * 6
        self.left_effort_pub.publish(msg)

    def set_right_impedance_params(self, diag_k: List[float], diag_d: List[float], diag_m: List[float]) -> bool:
        """Set impedance parameters on the right arm controller. Returns True on success."""
        client = AsyncParameterClient(self, "/ar2/cartesian_impedance_controller")
        if not client.wait_for_services(timeout_sec=5.0):
            self.get_logger().error("Could not reach /ar2/cartesian_impedance_controller parameter services")
            return False

        params = [
            Parameter("stiffness", Parameter.Type.DOUBLE_ARRAY, diag_k),
            Parameter("damping", Parameter.Type.DOUBLE_ARRAY, diag_d),
            Parameter("mass_matrix", Parameter.Type.DOUBLE_ARRAY, diag_m),
            Parameter("ft_force", Parameter.Type.DOUBLE_ARRAY, [0.0] * 6),
            Parameter("target_force", Parameter.Type.DOUBLE_ARRAY, [0.0] * 6),
        ]
        fut = client.set_parameters(params)
        while rclpy.ok() and not fut.done():
            rclpy.spin_once(self, timeout_sec=0.05)

        if not fut.done() or fut.result() is None:
            self.get_logger().error("Parameter set timed out or returned None")
            return False

        for r in fut.result().results:
            if not r.successful:
                self.get_logger().error(f"Parameter rejected: {r.reason}")
                return False

        self.get_logger().info("Impedance parameters set successfully")
        return True

    def send_right_goal(self, q_target: np.ndarray, dt: float) -> bool:
        if not self.right_action.wait_for_server(timeout_sec=2.0):
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q_target]
        pt.time_from_start.nanosec = int(max(dt, 0.02) * 1e9)
        goal.trajectory.points.append(pt)

        send_future = self.right_action.send_goal_async(goal)
        start = time.time()
        while rclpy.ok() and not send_future.done() and (time.time() - start < 3.0):
            rclpy.spin_once(self, timeout_sec=0.02)
        if not send_future.done() or send_future.result() is None:
            return False
        handle = send_future.result()
        if not handle.accepted:
            return False

        result_future = handle.get_result_async()
        end_t = time.time() + max(2.0 * dt, 0.1)
        while rclpy.ok() and not result_future.done() and time.time() < end_t:
            rclpy.spin_once(self, timeout_sec=0.02)
        return result_future.done()


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(cfg: Dict, task_name: str, dry_run: bool, rate_override: Optional[float]) -> int:
    runtime = cfg["runtime"]
    task = cfg["tasks"][task_name]
    robot_cfg = cfg["robot"]
    limits_cfg = cfg["limits"]

    runtime_cfg = RuntimeConfig(
        rate_hz=float(rate_override if rate_override else runtime["rate_hz"]),
        output_csv=str(runtime["output_csv"]),
        warmup_timeout_sec=float(runtime["warmup_timeout_sec"]),
        sensor_blend_alpha=float(runtime.get("sensor_blend_alpha", 0.0)),
    )

    left_base = np.array(robot_cfg["left_base_xyz"], dtype=float)
    right_base = np.array(robot_cfg["right_base_xyz"], dtype=float)
    ee_offset = np.array(robot_cfg["ee_offset_xyz"], dtype=float)

    # Optional base rotation (RPY)
    left_base_rpy = np.array(robot_cfg["left_base_rpy"], dtype=float) if "left_base_rpy" in robot_cfg else None
    right_base_rpy = np.array(robot_cfg["right_base_rpy"], dtype=float) if "right_base_rpy" in robot_cfg else None

    model = build_dual_arm_model(left_base, right_base, ee_offset, left_base_rpy, right_base_rpy)
    gains = ImpedanceGains(
        k_imp=np.diag(np.array(task["k_imp_diag"], dtype=float)),
        d_imp=np.diag(np.array(task["d_imp_diag"], dtype=float)),
        m_imp=np.diag(np.array(task["m_imp_diag"], dtype=float)),
    )
    limits = IntegratorLimits(
        tau_max=float(limits_cfg["tau_max"]),
        dq_max=float(limits_cfg["dq_max"]),
        ddq_max=float(limits_cfg["ddq_max"]),
    )

    rclpy.init()
    node = DualArmGeometricNode(cfg, task_name, dry_run)

    try:
        if not node.wait_for_joint_states(runtime_cfg.warmup_timeout_sec):
            node.get_logger().error("Timed out waiting for /joint_states and /ar2/joint_states")
            return 2

        if not node.switch_required_controllers():
            node.get_logger().error("Controller switching failed")
            return 3

        if not node.set_right_impedance_params(task["k_imp_diag"], task["d_imp_diag"], task["m_imp_diag"]):
            node.get_logger().error("Failed to set impedance parameters — aborting")
            return 5

        q_passive_0 = node.get_q(node.left_joint_map)
        q_active_0 = node.get_q(node.right_joint_map)
        if q_passive_0 is None or q_active_0 is None:
            node.get_logger().error("Initial joint state extraction failed")
            return 4

        state = ClosedLoopState(
            q_active=q_active_0.copy(),
            q_passive=q_passive_0.copy(),
            dq_active=np.zeros(6),
            dq_passive=np.zeros(6),
        )

        fk_active = forward_kinematics(state.q_active, model.right)
        fk_passive = forward_kinematics(state.q_passive, model.left)
        r_rel = fk_passive.ee_rot.T @ fk_active.ee_rot

        reduced = ReducedDynamics(model)
        controller = ImpedanceController(gains)
        integrator = GeometricIntegrator(model, reduced, controller, limits)

        dt = float(task["dt"])
        t0 = float(task["t0"])
        tf = float(task["tf"])

        # Use MEASURED passive position as trajectory start (not YAML),
        # to avoid initial error transients.
        q_passive_start_actual = q_passive_0.copy()
        q_passive_goal = np.array(task["q_passive_goal"], dtype=float)

        alpha = runtime_cfg.sensor_blend_alpha

        rows = []
        t = t0
        loop_period = dt  # Unified: integration step = loop period
        while rclpy.ok() and t <= tf + 1e-9:
            loop_start = time.time()

            # Trajectory generation using measured start position
            qd, dqd, ddqd = quintic_trajectory(
                t, t0, tf,
                q_passive_start_actual,
                q_passive_goal,
            )

            ref = TrajectoryReference(qd_passive=qd, dqd_passive=dqd, ddqd_passive=ddqd)
            out = integrator.step(state, ref, dt, r_rel)
            state = out.state

            # Sensor feedback fusion (complementary filter)
            if alpha > 0.0:
                rclpy.spin_once(node, timeout_sec=0.0)
                q_passive_meas = node.get_q(node.left_joint_map)
                q_active_meas = node.get_q(node.right_joint_map)
                if q_passive_meas is not None and q_active_meas is not None:
                    state = ClosedLoopState(
                        q_active=(1.0 - alpha) * state.q_active + alpha * q_active_meas,
                        q_passive=(1.0 - alpha) * state.q_passive + alpha * q_passive_meas,
                        dq_active=state.dq_active,
                        dq_passive=state.dq_passive,
                    )

            # Publish commands
            node.publish_left_gravity_hold()
            if not dry_run:
                ok = node.send_right_goal(state.q_active, dt)
                if not ok:
                    node.get_logger().warn("Right-arm action goal did not finish in allotted time")

            # Safety check
            if out.diagnostics["constraint_pos_error"] > 0.1:
                node.get_logger().error(
                    f"SAFETY: Constraint position error {out.diagnostics['constraint_pos_error']:.4f} > 0.1m, stopping!"
                )
                break

            # Log
            row = {
                "t": float(t),
                "constraint_pos_error": out.diagnostics["constraint_pos_error"],
                "constraint_rot_error": out.diagnostics["constraint_rot_error"],
                "ik_ok": out.diagnostics["ik_ok"],
                "ik_residual": out.diagnostics["ik_residual"],
                "tau_norm": out.diagnostics["tau_norm"],
                "manipulability": out.diagnostics["manipulability"],
                "jcn_norm": out.diagnostics["jcn_norm"],
                "mr_min_eig": out.diagnostics["mr_min_eig"],
            }
            for i in range(6):
                row[f"q_active_{i+1}"] = float(state.q_active[i])
                row[f"q_passive_{i+1}"] = float(state.q_passive[i])
                row[f"tau_active_{i+1}"] = float(out.tau_active[i])
            rows.append(row)

            t += dt
            rclpy.spin_once(node, timeout_sec=0.0)

            # Sleep for remainder of loop period (unified with dt)
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, loop_period - elapsed))

        if rows:
            out_path = Path(runtime_cfg.output_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            node.get_logger().info(f"Saved log: {out_path}")

    finally:
        node.destroy_node()
        rclpy.shutdown()

    return 0


def parse_args():
    p = argparse.ArgumentParser(description="Dual-arm geometric reduced closed-loop controller")
    default_cfg = Path(__file__).with_name("dual_arm_geometric_config.yaml")
    p.add_argument("--config", default=str(default_cfg))
    p.add_argument("--task", default="push")
    p.add_argument("--rate", type=float, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.task not in cfg.get("tasks", {}):
        print(f"Unknown task '{args.task}'. Available: {list(cfg.get('tasks', {}).keys())}", file=sys.stderr)
        return 1
    return run(cfg, args.task, args.dry_run, args.rate)


if __name__ == "__main__":
    raise SystemExit(main())
