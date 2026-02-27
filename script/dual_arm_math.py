"""Math stack for reduced closed-loop dual-arm geometric control.

Implements the algorithm blocks from the paper pseudocode (Algorithm 1)
and equations (10)-(22):
  1) RNEA-based dynamics:  M(q), C(q,dq)*dq + G(q)  per arm
  2) Reduced dynamics:     Mr, CGr (with dN term), Gr, S, dS
  3) Impedance control:    tau_I = (S^T)^+ * u     (Eq. 21-22)
  4) Symplectic update:    velocity-first Euler     (Alg. 1, Step 3)
  5) Retraction map:       FK + IK                  (Alg. 1, Step 4)

Ported from paper_code/script/{recursive_NE.py, TMT.py, reduced_dynamics.py,
controller.py, integrator.py}, adapted for the Addverb Heal cobot model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.linalg import block_diag

from dual_arm_addverb_model import DualArmModel, RobotModel


# ---------------------------------------------------------------------------
# Basic math helpers
# ---------------------------------------------------------------------------

def skew(v: np.ndarray) -> np.ndarray:
    """3-vector → 3×3 skew-symmetric matrix."""
    if v.ndim == 2:
        v = v.reshape(-1)
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def axis_angle_to_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' formula: axis (unit or not) + angle → 3×3 rotation."""
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    axis = axis / n
    k = skew(axis)
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def damped_pinv(mat: np.ndarray, damping: float = 1e-3) -> np.ndarray:
    """Damped Moore-Penrose pseudo-inverse."""
    r, c = mat.shape
    if r <= c:
        return mat.T @ np.linalg.inv(mat @ mat.T + (damping ** 2) * np.eye(r))
    return np.linalg.inv(mat.T @ mat + (damping ** 2) * np.eye(c)) @ mat.T


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FKResult:
    ee_pos: np.ndarray
    ee_rot: np.ndarray
    joint_positions: np.ndarray      # (n, 3) — each joint origin in world
    joint_axes_world: np.ndarray     # (n, 3) — each joint axis in world
    link_transforms: list            # list of (pre_rot_4x4, post_rot_4x4) per link


@dataclass
class ClosedLoopState:
    q_active: np.ndarray
    q_passive: np.ndarray
    dq_active: np.ndarray
    dq_passive: np.ndarray


@dataclass
class TrajectoryReference:
    qd_passive: np.ndarray
    dqd_passive: np.ndarray
    ddqd_passive: np.ndarray


@dataclass
class ImpedanceGains:
    k_imp: np.ndarray
    d_imp: np.ndarray
    m_imp: np.ndarray


@dataclass
class IntegratorLimits:
    tau_max: float
    dq_max: float
    ddq_max: float


@dataclass
class StepOutput:
    state: ClosedLoopState
    diagnostics: Dict[str, float]
    tau_active: np.ndarray


# ---------------------------------------------------------------------------
# Forward Kinematics
# ---------------------------------------------------------------------------

def forward_kinematics(q: np.ndarray, robot: RobotModel) -> FKResult:
    """Compute FK chain for all joints and the end-effector.

    Builds 4×4 homogeneous transforms through the chain, storing both
    the pre-joint-rotation frame (for joint position/axis) and
    post-rotation frame (for link CoM / inertia in world).
    """
    t = np.eye(4)
    t[:3, 3] = robot.base_xyz
    t[:3, :3] = robot.base_rot

    joint_positions = []
    joint_axes = []
    link_transforms = []

    for i, link in enumerate(robot.links):
        # Apply joint origin offset
        t_origin = np.eye(4)
        t_origin[:3, 3] = link.joint_origin_xyz
        t = t @ t_origin

        # Pre-rotation frame: joint position and axis are here
        pre_rot = t.copy()
        joint_positions.append(t[:3, 3].copy())
        axis_world = t[:3, :3] @ link.joint_axis
        nrm = np.linalg.norm(axis_world)
        if nrm > 1e-12:
            axis_world = axis_world / nrm
        joint_axes.append(axis_world)

        # Apply joint rotation
        t_rot = np.eye(4)
        t_rot[:3, :3] = axis_angle_to_rot(link.joint_axis, q[i])
        t = t @ t_rot

        link_transforms.append((pre_rot, t.copy()))

    # End-effector
    t_ee = np.eye(4)
    t_ee[:3, 3] = robot.ee_offset_xyz
    t = t @ t_ee

    return FKResult(
        ee_pos=t[:3, 3].copy(),
        ee_rot=t[:3, :3].copy(),
        joint_positions=np.array(joint_positions),
        joint_axes_world=np.array(joint_axes),
        link_transforms=link_transforms,
    )


# ---------------------------------------------------------------------------
# Jacobian (end-effector)
# ---------------------------------------------------------------------------

def jacobian(q: np.ndarray, robot: RobotModel) -> np.ndarray:
    """6×n geometric Jacobian for the end-effector: [Jv; Jw]."""
    fk = forward_kinematics(q, robot)
    n = len(q)
    jv = np.zeros((3, n))
    jw = np.zeros((3, n))
    for i in range(n):
        z = fk.joint_axes_world[i]
        p = fk.joint_positions[i]
        jv[:, i] = np.cross(z, fk.ee_pos - p)
        jw[:, i] = z
    return np.vstack((jv, jw))


# ---------------------------------------------------------------------------
# CoM Jacobians (for TMT Mass Matrix) — ported from paper_code/script/TMT.py
# ---------------------------------------------------------------------------

def _com_jacobians(q: np.ndarray, robot: RobotModel) -> list:
    """Compute 6×n Jacobian for each link's centre of mass.

    Returns list of n matrices, each (6, n).
    Ported from paper_code/script/TMT.py → jac_torques_generalized().
    """
    fk = forward_kinematics(q, robot)
    n = len(q)
    j_list = []

    for link_idx in range(n):
        _, post_rot = fk.link_transforms[link_idx]
        # CoM in world frame
        com_world = post_rot[:3, :3] @ robot.links[link_idx].com_xyz + post_rot[:3, 3]

        jv_cols = []
        jw_cols = []
        for joint_idx in range(n):
            if joint_idx <= link_idx:
                z_j = fk.joint_axes_world[joint_idx]
                o_j = fk.joint_positions[joint_idx]
                jv_col = np.cross(z_j, com_world - o_j)
                jw_col = z_j
            else:
                jv_col = np.zeros(3)
                jw_col = np.zeros(3)
            jv_cols.append(jv_col)
            jw_cols.append(jw_col)

        jv = np.column_stack(jv_cols)
        jw = np.column_stack(jw_cols)
        j_list.append(np.vstack([jv, jw]))

    return j_list


# ---------------------------------------------------------------------------
# Mass matrix via TMT — ported from paper_code/script/TMT.py → TMT()
# ---------------------------------------------------------------------------

def mass_matrix_tmt(q: np.ndarray, robot: RobotModel) -> np.ndarray:
    """Configuration-dependent n×n mass matrix M(q) via the TMT method.

    M = sum_i  J_gi^T * M_spatial_i * J_gi
    where J_gi is the CoM Jacobian and M_spatial_i = diag(m_i*I3, I_i_world).
    """
    fk = forward_kinematics(q, robot)
    j_list = _com_jacobians(q, robot)
    n = len(q)

    m_blocks = []
    for i in range(n):
        link = robot.links[i]
        _, post_rot = fk.link_transforms[i]
        R = post_rot[:3, :3]

        m = max(link.mass, 1e-6)
        mass_3x3 = m * np.eye(3)

        # Inertia in world frame: R * I_body * R^T
        I_world = R @ link.inertia_matrix @ R.T

        m_blocks.append(block_diag(mass_3x3, I_world))

    M_spatial = block_diag(*m_blocks)
    J_all = np.vstack(j_list)
    M = J_all.T @ M_spatial @ J_all

    # Symmetrise (eliminate numerical asymmetry)
    return 0.5 * (M + M.T)


# ---------------------------------------------------------------------------
# RNEA — ported from paper_code/script/recursive_NE.py
# ---------------------------------------------------------------------------

class RNEASolver:
    """Recursive Newton-Euler Algorithm adapted for RobotModel (Addverb cobot).

    Computes the inverse dynamics: tau = M(q)*ddq + C(q,dq)*dq + G(q).
    By setting specific inputs to zero, we extract individual terms.
    """

    def __init__(self, gravity: np.ndarray = np.array([0.0, 0.0, -9.81])):
        self.gravity = gravity.copy()

    def compute_id(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray,
                   robot: RobotModel) -> np.ndarray:
        """Full inverse dynamics: returns n-dim torque vector."""
        fk = forward_kinematics(q, robot)
        n = len(q)
        g_vec = self.gravity

        # ----- Forward recursion -----
        omega = [None] * n        # angular velocity in link frame
        v_lin = [None] * n        # linear velocity of joint origin in link frame
        omega_dot = [None] * n    # angular acceleration in link frame
        v_dot = [None] * n        # linear acceleration of joint origin in link frame
        c_ddot = [None] * n       # linear acceleration of CoM in link frame

        omega_prev = np.zeros(3)
        v_prev = np.zeros(3)
        omega_dot_prev = np.zeros(3)
        v_dot_prev = np.zeros(3)

        for i in range(n):
            link = robot.links[i]
            pre_rot, post_rot = fk.link_transforms[i]

            # R from parent frame to current frame (transpose of local rotation)
            if i == 0:
                # First link: parent is base
                R_base = robot.base_rot
                R_joint = axis_angle_to_rot(link.joint_axis, q[i])
                # Origin offset in parent frame
                O_i = link.joint_origin_xyz
                # R_{i}^{i-1} = (R_base * R_origin * R_joint)^T * R_base * R_origin
                # Simplified: R_local = R_joint for revolute joints with identity origin rotation
                R_to_local = R_joint.T  # transform parent velocities to local frame
            else:
                R_joint = axis_angle_to_rot(link.joint_axis, q[i])
                O_i = link.joint_origin_xyz
                R_to_local = R_joint.T

            z_i = link.joint_axis  # joint axis in local frame
            b_i = link.com_xyz     # CoM in local frame

            omega[i] = R_to_local @ omega_prev + z_i * dq[i]
            v_lin[i] = R_to_local @ (v_prev + np.cross(omega_prev, O_i))
            omega_dot[i] = (R_to_local @ omega_dot_prev
                            + np.cross(R_to_local @ omega_prev, z_i * dq[i])
                            + z_i * ddq[i])
            v_dot[i] = R_to_local @ (v_dot_prev
                                     + np.cross(omega_dot_prev, O_i)
                                     + np.cross(omega_prev, np.cross(omega_prev, O_i)))
            c_ddot[i] = (v_dot[i]
                         + np.cross(omega_dot[i], b_i)
                         + np.cross(omega[i], np.cross(omega[i], b_i)))

            omega_prev = omega[i]
            v_prev = v_lin[i]
            omega_dot_prev = omega_dot[i]
            v_dot_prev = v_dot[i]

        # ----- Backward recursion -----
        f_next = np.zeros(3)
        n_next = np.zeros(3)

        tau = np.zeros(n)
        # O for next link (end-effector offset for last link)
        O_next = robot.ee_offset_xyz.copy()

        for j in range(n - 1, -1, -1):
            link = robot.links[j]
            _, post_rot = fk.link_transforms[j]
            R_world = post_rot[:3, :3]

            m = max(link.mass, 1e-6)
            I_body = link.inertia_matrix.copy()
            b_j = link.com_xyz
            r_j = O_next - b_j

            # Gravity in body frame
            g_body = R_world.T @ g_vec

            # Inertial force and torque (D'Alembert)
            F = m * c_ddot[j]
            N = I_body @ omega_dot[j] + np.cross(omega[j], I_body @ omega[j])

            # Transform f_next / n_next from child frame to current frame
            R_child = axis_angle_to_rot(
                robot.links[j + 1].joint_axis if j + 1 < n else np.array([0, 0, 1]),
                0.0  # identity rotation for the "frame" part
            )
            # In standard RNEA, we use the rotation from i+1 to i.
            # With our FK structure: child's pre_rot relative to parent's post_rot.
            if j + 1 < n:
                R_child_to_parent = axis_angle_to_rot(robot.links[j + 1].joint_axis, 0.0)
                # This is identity since we're just at the origin
                # The actual rotation is handled by the joint. In the
                # backward pass, f and n are in the current link's frame,
                # and we transform from child link frame.
                # For simplicity, use the approach: accumulate in world frame.
                pass

            # Simpler approach: compute everything in the world frame
            f_j = F + f_next - m * g_body
            n_j = (N + n_next
                   + np.cross(b_j, f_j)
                   + np.cross(r_j, f_next))

            # Project onto joint axis to get torque
            tau[j] = np.dot(link.joint_axis, n_j)

            f_next = f_j
            n_next = n_j
            if j > 0:
                # Transform to parent frame
                R_joint_inv = axis_angle_to_rot(link.joint_axis, q[j]).T
                f_next = R_joint_inv @ f_next
                n_next = R_joint_inv @ n_next
                O_next = link.joint_origin_xyz

        return tau

    def compute_mcg(self, q: np.ndarray, dq: np.ndarray,
                    robot: RobotModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute M(q), CG(q,dq), G(q) for one arm.

        Returns:
            M:  (n, n) mass matrix
            CG: (n, 1) Coriolis+Gravity vector  C(q,dq)*dq + G(q)
            G:  (n, 1) pure gravity vector  G(q)
        """
        n = len(q)

        # M via TMT (more robust than column-by-column RNEA)
        M = mass_matrix_tmt(q, robot)

        # G = RNEA(q, dq=0, ddq=0)
        G = self.compute_id(q, np.zeros(n), np.zeros(n), robot)

        # CG = RNEA(q, dq, ddq=0) = C(q,dq)*dq + G(q)
        CG = self.compute_id(q, dq, np.zeros(n), robot)

        return M, CG[:, np.newaxis], G[:, np.newaxis]


# ---------------------------------------------------------------------------
# Reduced Dynamics — ported from paper_code/script/reduced_dynamics.py
# ---------------------------------------------------------------------------

class ReducedDynamics:
    """Reduced-order dynamics on the constraint manifold (Eq. 10).

    Computes Mr, CGr, Gr, S following the paper's formulation exactly.
    """

    def __init__(self, model: DualArmModel):
        self.model = model
        self.rnea_active = RNEASolver()
        self.rnea_passive = RNEASolver()

    def s_matrix(self, q_active: np.ndarray, q_passive: np.ndarray) -> np.ndarray:
        """Kinematic coupling: S = J_I^† J_II  (Eq. 7)."""
        ja = jacobian(q_active, self.model.right)
        jp = jacobian(q_passive, self.model.left)
        # Use plain pinv matching the paper (reduced_dynamics.py line 19),
        # with fallback to damped pinv near singularity.
        w = np.sqrt(max(np.linalg.det(ja @ ja.T), 0.0))
        if w < 1e-6:
            return damped_pinv(ja, 1e-2) @ jp
        return np.linalg.pinv(ja) @ jp

    def ds_matrix(self, q_active: np.ndarray, q_passive: np.ndarray,
                  dq_active: np.ndarray, dq_passive: np.ndarray,
                  eps: float = 1e-6) -> np.ndarray:
        """Numerical Ṡ via central differences — matches paper exactly.

        paper_code/script/reduced_dynamics.py lines 22-32.
        """
        s_next = self.s_matrix(
            q_active + dq_active * eps,
            q_passive + dq_passive * eps,
        )
        s_prev = self.s_matrix(
            q_active - dq_active * eps,
            q_passive - dq_passive * eps,
        )
        return (s_next - s_prev) / (2.0 * eps)

    def reduced_terms(
        self,
        q_active: np.ndarray,
        q_passive: np.ndarray,
        dq_active: np.ndarray,
        dq_passive: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute reduced dynamics matrices.

        Returns:
            mr:  (m, m) reduced mass matrix — Eq. 10
            cgr: (m, 1) reduced Coriolis+Gravity — Eq. 10 (includes dN term)
            gr:  (m, 1) reduced pure gravity — Eq. 10
            s:   (n, m) coupling matrix S
        """
        # Per-arm dynamics via RNEA
        m_active, cg_active, g_active = self.rnea_active.compute_mcg(
            q_active, dq_active, self.model.right
        )
        m_passive, cg_passive, g_passive = self.rnea_passive.compute_mcg(
            q_passive, dq_passive, self.model.left
        )

        s = self.s_matrix(q_active, q_passive)
        n = len(q_active)
        m = len(q_passive)

        # N matrix (Eq. 8): N = [S; I_m]
        n_matrix = np.vstack((s, np.eye(m)))

        # Full block-diagonal mass matrix
        m_full = np.block([
            [m_active, np.zeros((n, m))],
            [np.zeros((m, n)), m_passive],
        ])

        # Mr = N^T M N  (Eq. 10)
        mr = n_matrix.T @ m_full @ n_matrix
        mr = 0.5 * (mr + mr.T)  # symmetrise

        # dN matrix: dN = [dS; 0]
        ds = self.ds_matrix(q_active, q_passive, dq_active, dq_passive)
        dn_matrix = np.vstack((ds, np.zeros((m, m))))

        # CGr = N^T (CG_full + M_full * dN * dq_passive)  (Eq. 10)
        cg_full = np.vstack((cg_active, cg_passive))
        cgr = n_matrix.T @ (cg_full + m_full @ (dn_matrix @ dq_passive[:, np.newaxis]))

        # Gr = S^T G_active + G_passive  (Eq. 10)
        gr = s.T @ g_active + g_passive

        return mr, cgr, gr, s


# ---------------------------------------------------------------------------
# Impedance Controller — ported from paper_code/script/controller.py
# ---------------------------------------------------------------------------

class ImpedanceController:
    """Joint-space impedance control on the constraint manifold (Eq. 21-22).

    u = Mr * [ddq_d - M_imp^{-1}(K_imp*e + D_imp*ed)] + CGr
    tau_I = (S^T)^† * u
    """

    def __init__(self, gains: ImpedanceGains, damping_lambda: float = 0.025):
        self.gains = gains
        self.damping_lambda = damping_lambda
        self.m_imp_inv = np.linalg.pinv(gains.m_imp)

    def compute_tau(
        self,
        q_passive: np.ndarray,
        dq_passive: np.ndarray,
        reference: TrajectoryReference,
        mr: np.ndarray,
        cgr: np.ndarray,
        s: np.ndarray,
    ) -> np.ndarray:
        """Compute active torque tau_I (Eq. 21-22).

        Note: cgr = C_r*dq_II + G_r  (combined Coriolis+Gravity from RNEA).
        This matches the paper's controller.py line 43: u = mr @ acc_cmd + cgr
        """
        e = (q_passive - reference.qd_passive)[:, np.newaxis]
        ed = (dq_passive - reference.dqd_passive)[:, np.newaxis]

        acc_cmd = reference.ddqd_passive[:, np.newaxis] - self.m_imp_inv @ (
            self.gains.k_imp @ e + self.gains.d_imp @ ed
        )

        # Eq. 21: u = Mr * acc_cmd + CGr
        u = mr @ acc_cmd + cgr

        # Eq. 22: tau_I = (S^T)^† * u
        s_t = s.T
        s_t_pinv = damped_pinv(s_t, self.damping_lambda)
        tau = s_t_pinv @ u
        return tau.reshape(-1)


# ---------------------------------------------------------------------------
# IK Solver
# ---------------------------------------------------------------------------

def orientation_error(current_r: np.ndarray, target_r: np.ndarray) -> np.ndarray:
    """Orientation error via cross-product sum (same as paper)."""
    return 0.5 * (
        np.cross(current_r[:, 0], target_r[:, 0])
        + np.cross(current_r[:, 1], target_r[:, 1])
        + np.cross(current_r[:, 2], target_r[:, 2])
    )


def solve_active_ik(
    q_seed: np.ndarray,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    robot: RobotModel,
    max_iters: int = 100,
    tol: float = 2e-3,
    damping: float = 2e-2,
) -> Tuple[np.ndarray, bool, float]:
    """Iterative IK via damped Newton-Raphson with joint limit clamping.

    Uses the same error convention as the paper (integrator.py line 53):
    pos_err = current - target  →  dq = -J†*err  (standard Newton-Raphson).
    """
    q = q_seed.copy()
    for _ in range(max_iters):
        fk = forward_kinematics(q, robot)
        pos_err = fk.ee_pos - target_pos
        rot_err = orientation_error(fk.ee_rot, target_rot)
        err = np.concatenate((pos_err, rot_err))
        nerr = float(np.linalg.norm(err))
        if nerr < tol:
            return q, True, nerr

        j = jacobian(q, robot)
        dq = -damped_pinv(j, damping) @ err
        q = q + dq

        # Clamp to URDF joint limits
        for i, link in enumerate(robot.links):
            q[i] = np.clip(q[i], link.q_min, link.q_max)

    fk = forward_kinematics(q, robot)
    res = np.concatenate((fk.ee_pos - target_pos, orientation_error(fk.ee_rot, target_rot)))
    return q, False, float(np.linalg.norm(res))


# ---------------------------------------------------------------------------
# Quintic Trajectory
# ---------------------------------------------------------------------------

def quintic_trajectory(t: float, t0: float, tf: float,
                       p0: np.ndarray, pf: np.ndarray):
    """5th-order polynomial with zero velocity/acceleration at endpoints."""
    if t <= t0:
        return p0.copy(), np.zeros_like(p0), np.zeros_like(p0)
    if t >= tf:
        return pf.copy(), np.zeros_like(pf), np.zeros_like(pf)

    tau = (t - t0) / (tf - t0)
    tau2 = tau * tau
    tau3 = tau2 * tau
    tau4 = tau3 * tau
    tau5 = tau4 * tau

    s = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5
    ds_dtau = 30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4
    d2s_dtau2 = 60.0 * tau - 180.0 * tau2 + 120.0 * tau3

    scale = 1.0 / (tf - t0)
    q = p0 + s * (pf - p0)
    dq = ds_dtau * scale * (pf - p0)
    ddq = d2s_dtau2 * (scale ** 2) * (pf - p0)
    return q, dq, ddq


# ---------------------------------------------------------------------------
# Geometric Integrator — ported from paper_code/script/integrator.py
# ---------------------------------------------------------------------------

class GeometricIntegrator:
    """Algorithm 1: Geometric Simulation of Reduced Closed-Loop Dynamics.

    Integrates the reduced system on the constraint manifold M using
    symplectic Euler + exact retraction via IK.
    """

    def __init__(
        self,
        model: DualArmModel,
        reduced_dynamics: ReducedDynamics,
        impedance_controller: ImpedanceController,
        limits: IntegratorLimits,
    ):
        self.model = model
        self.reduced_dynamics = reduced_dynamics
        self.impedance_controller = impedance_controller
        self.limits = limits
        self.last_valid_q_active = None

    def step(
        self,
        state: ClosedLoopState,
        reference: TrajectoryReference,
        dt: float,
        r_rel: np.ndarray,
    ) -> StepOutput:
        # === Step 1: Control Formulation (Alg. 1, lines 4-7) ===
        mr, cgr, gr, s = self.reduced_dynamics.reduced_terms(
            state.q_active, state.q_passive, state.dq_active, state.dq_passive
        )

        # Compute tau_I via impedance law (Eq. 21-22)
        # cgr already contains C_r*dq + G_r (combined), matching paper
        tau_active = self.impedance_controller.compute_tau(
            state.q_passive, state.dq_passive, reference, mr, cgr, s
        )
        tau_active = np.clip(tau_active, -self.limits.tau_max, self.limits.tau_max)

        # === Step 2: Reduced Acceleration (Alg. 1, line 9) ===
        # ddq_II = Mr^{-1} (S^T tau_I - CGr)
        # Note: cgr = C_r*dq + G_r (combined), so we subtract cgr
        # This matches paper: integrator.py line 94: solve(mr, reduced_force - cgr)
        reduced_force = s.T @ tau_active[:, np.newaxis]
        try:
            ddq_passive = np.linalg.solve(mr, reduced_force - cgr).reshape(-1)
        except np.linalg.LinAlgError:
            ddq_passive = (np.linalg.pinv(mr) @ (reduced_force - cgr)).reshape(-1)

        ddq_passive = np.clip(ddq_passive, -self.limits.ddq_max, self.limits.ddq_max)

        # === Step 3: Symplectic Update (Alg. 1, lines 11-12) ===
        dq_passive_new = np.clip(
            state.dq_passive + ddq_passive * dt,
            -self.limits.dq_max, self.limits.dq_max
        )
        q_passive_new = state.q_passive + dq_passive_new * dt

        # === Step 4: Retraction Map (Alg. 1, lines 14-18) ===
        # 4a. Dependent velocity (line 14)
        dq_active_new = (s @ dq_passive_new[:, np.newaxis]).reshape(-1)

        # 4b. FK target from passive arm (lines 15-16)
        fk_passive = forward_kinematics(q_passive_new, self.model.left)
        target_pos = fk_passive.ee_pos
        target_rot = fk_passive.ee_rot @ r_rel

        # 4c. NaN/Inf guard (from paper integrator.py lines 109-116)
        if (np.any(~np.isfinite(state.q_active))
                or np.any(~np.isfinite(q_passive_new))
                or np.any(~np.isfinite(dq_passive_new))):
            q_active_new = (self.last_valid_q_active.copy()
                            if self.last_valid_q_active is not None
                            else state.q_active.copy())
            ik_ok = False
            ik_res = float("inf")
        else:
            # 4d. Exact state lift via IK (line 18)
            q_active_new, ik_ok, ik_res = solve_active_ik(
                state.q_active,
                target_pos,
                target_rot,
                self.model.right,
            )

        if not ik_ok:
            if self.last_valid_q_active is not None:
                q_active_new = self.last_valid_q_active.copy()
            else:
                q_active_new = state.q_active + dq_active_new * dt
        else:
            self.last_valid_q_active = q_active_new.copy()

        # Recompute S at new configuration for velocity (Phase 3 improvement)
        s_new = self.reduced_dynamics.s_matrix(q_active_new, q_passive_new)
        dq_active_new = (s_new @ dq_passive_new[:, np.newaxis]).reshape(-1)

        # === Step 5: State Update (Alg. 1, lines 20-21) ===
        fk_active = forward_kinematics(q_active_new, self.model.right)
        pos_err = np.linalg.norm(fk_active.ee_pos - fk_passive.ee_pos)
        rot_err = np.linalg.norm(fk_active.ee_rot - target_rot)

        # Manipulability diagnostic
        ja = jacobian(state.q_active, self.model.right)
        manipulability = np.sqrt(max(np.linalg.det(ja @ ja.T), 0.0))

        # Constraint consistency check: ||Jc * N|| (from paper integrator.py)
        jp = jacobian(q_passive_new, self.model.left)
        n_matrix = np.vstack((s, np.eye(len(q_passive_new))))
        jc_matrix = np.hstack((ja, -jp))
        jcn_norm = float(np.linalg.norm(jc_matrix @ n_matrix))

        new_state = ClosedLoopState(
            q_active=q_active_new,
            q_passive=q_passive_new,
            dq_active=dq_active_new,
            dq_passive=dq_passive_new,
        )
        diag = {
            "constraint_pos_error": float(pos_err),
            "constraint_rot_error": float(rot_err),
            "ik_ok": float(1.0 if ik_ok else 0.0),
            "ik_residual": float(ik_res),
            "mr_symmetry": float(np.linalg.norm(mr - mr.T)),
            "mr_min_eig": float(np.min(np.linalg.eigvalsh((mr + mr.T) * 0.5))),
            "tau_norm": float(np.linalg.norm(tau_active)),
            "manipulability": float(manipulability),
            "jcn_norm": jcn_norm,
        }
        return StepOutput(state=new_state, diagnostics=diag, tau_active=tau_active)
