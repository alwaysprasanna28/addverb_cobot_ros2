# Implementation Plan: Aligning Code with Paper Algorithm (Revised)

**Paper**: *Geometric Simulation of Reduced Closed-Loop Dynamics* (dual_arm_robot_control.pdf)  
**Reference Implementation**: `paper_code/script/` — UR5e / MuJoCo (the working original)  
**Target Implementation**: `cobot_ros2/script/` — Addverb Heal Cobot / ROS2  
**Hardware Docs**: `cobot_ros2/control.md`, URDF `heal.urdf.xacro`, ros2_control `heal.ros2_control.xacro`  
**Date**: 2026-02-27  

---

## Key Findings from Hardware + Reference Code Deep-Dive

Before diving into the plan, here are critical facts discovered by reading the actual source code:

### 1. Paper's Reference Code Uses Full RNEA Dynamics (not heuristics)
The paper's `paper_code/script/` uses:
- **`TMT.py`** → Full Composite Rigid Body mass matrix via J^T M J (Transport Matrix method)
- **`recursive_NE.py`** → Full Recursive Newton-Euler for M(q), C(q,dq)*dq, G(q) per arm
- **`reduced_dynamics.py`** → Proper `Mr`, `CGr` (including `N^T (CG + M * dN * dq_passive)`), `Gr` = `S^T G_active + G_passive`

The ROS2 port (**`dual_arm_math.py`**) replaced ALL of this with: constant diagonal inertia, viscous friction, sin(q) gravity. This is the root cause of every dynamics issue.

### 2. Paper's Controller Passes `cgr` (which includes gravity) to `compute_control`
In the paper's `controller.py` line 43:
```python
u = mr @ acceleration_cmd + cgr  # cgr = C_r * dq + G_r (combined from RNEA)
```
The `cgr` returned by `RNEA.mcg_matrix()` is **CG** (Coriolis+Gravity combined as a single vector), NOT just Coriolis. This is because RNEA computes `C(q,dq)*dq + G(q)` as one pass (line 103: `CG = self.compute_matrix(q_dot, q_ddot=np.zeros, g=g)`).

**This means the paper's control law is**: `u = Mr * acc_cmd + CGr` where `CGr = C_r*dq_II + G_r`. The hr/gravity is already embedded in cgr.

### 3. Addverb `gravity_comp_effort_controller` Does NOT Add Gravity Internally
Reading `gravity_comp_effort_controller.cpp` (line 144):
```cpp
command_interfaces_[i].set_value(commanded_effort_[i]);
```
It simply **passes through** the commanded effort to the hardware. The name "gravity_comp" is misleading — the controller itself does NOT compute or add gravity compensation. It's a raw effort forwarder. The gravity compensation likely happens at the firmware/hardware level on the Addverb cobot.

### 4. Addverb `cartesian_impedance_controller` Interface
From `cartesian_impedance_controller.h` and `heal.ros2_control.xacro`:
- Uses `FollowJointTrajectory` action server (confirmed)
- Parameters: `stiffness` (6), `damping` (6), `mass_matrix` (6), `ft_force` (6), `target_force` (6)
- The impedance is computed **inside the hardware controller on the cobot**, not in our Python code
- Our code sends **joint position targets** via the action, and the controller does impedance tracking internally

### 5. URDF Confirmation
- Joint names: `joint1`–`joint6` ✅ (matches code)
- Joint axes match `dual_arm_addverb_model.py` ✅
- Joint origins match ✅
- Inertial parameters match ✅ (with one exception: URDF has off-diagonal inertia terms that the code's `inertia_diag` ignores)
- Joint 6 / `end-effector` link has **no inertial tag** in the URDF ✅ (explains zero mass/inertia in model)
- Joint limits are defined in URDF but not used in IK

---

## Revised Understanding: What Actually Needs Fixing

Given the above findings, the architecture is:

```
┌─────────────────────────────────────────────────────────────────┐
│ Python Script (our code)                                        │
│                                                                 │
│  Algorithm 1 loop:                                              │
│    1. Compute M_r, C_r*dq+G_r, S (BROKEN → heuristics)        │
│    2. Impedance control → τ_I                                   │
│    3. Reduced acceleration → ddq_II                             │
│    4. Symplectic Euler → q_II_new, dq_II_new                   │
│    5. Retraction: FK(q_II_new) → target, IK → q_I_new          │
│    6. Send q_I_new to hardware via FollowJointTrajectory action │
│    7. Send [0]*6 to left arm gravity_comp controller            │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│ Addverb Hardware (ros2_control)                                 │
│                                                                 │
│  Right arm: cartesian_impedance_controller                      │
│    - Receives joint position targets via action                 │
│    - Applies impedance control internally (K, D, M params)      │
│                                                                 │
│  Left arm: gravity_comp_effort_controller                       │
│    - Passes raw effort commands to hardware                     │
│    - "Gravity comp" is at firmware level                         │
└─────────────────────────────────────────────────────────────────┘
```

The fundamental fix is: **Port the paper's RNEA/TMT-based dynamics to work with Addverb model parameters**, replacing the heuristic approximations. The ROS2 interface is structurally correct but needs refinements.

---

## Phase 1: Port RNEA-Based Dynamics from Paper Code 
**Files**: `dual_arm_math.py` (major rewrite), `dual_arm_addverb_model.py` (model extensions)  
**Issues Resolved**: #1 (Coriolis), #2 (Gravity), #3 (Mass Matrix), #4 (Control Law), #5 (Ṡ)

This is the single most important phase. We port the paper's `recursive_NE.py` + `TMT.py` + `reduced_dynamics.py` to work with the Addverb robot model.

### Step 1.1: Add Full Inertia Data to `LinkSpec` (from URDF)

The URDF has **full 3×3 inertia tensors** (off-diagonal terms), but `dual_arm_addverb_model.py` only stores diagonal. Fix:

```python
@dataclass
class LinkSpec:
    joint_name: str
    joint_origin_xyz: np.ndarray
    joint_axis: np.ndarray
    com_xyz: np.ndarray          # CoM relative to link frame
    mass: float
    inertia_matrix: np.ndarray   # Full 3x3 inertia tensor (was: inertia_diag)
```

Update `_base_link_specs()` with the URDF's full inertia tensors:
```python
# Example for link1 (from heal.urdf.xacro lines 30-31):
# ixx=0.011934, ixy=-4.2772E-06, ixz=0.0016521
# iyy=0.011427, iyz=3.2849E-06, izz=0.0091785
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
),
# ... repeat for all links with URDF values ...
```

### Step 1.2: Add Joint Limits to `LinkSpec` (from URDF)

```python
@dataclass
class LinkSpec:
    # ... existing fields ...
    q_min: float = -np.pi    # Joint lower limit
    q_max: float = np.pi     # Joint upper limit
```

Populate from URDF:
- joint1: [-π, π]
- joint2: [-0.75, 2.05]
- joint3: [-1.0, 1.0]
- joint4: [-π, π]
- joint5: [-π, π]
- joint6: [-π, π]

### Step 1.3: Implement RNEA for Addverb Cobot

Port `paper_code/script/recursive_NE.py` and `TMT.py` into a new class in `dual_arm_math.py`. The key function is `mcg_matrix(q, dq, robot)` which returns `(M, CG, G)`:

- **M(q)**: 6×6 configuration-dependent mass matrix (via TMT method: `J_g^T * M_spatial * J_g`)
- **CG(q,dq)**: 6×1 Coriolis+Gravity combined vector (RNEA with `q̈=0`)
- **G(q)**: 6×1 pure gravity vector (RNEA with `q̇=0, q̈=0`)

The implementation must:
1. Use `forward_kinematics()` already in `dual_arm_math.py` for the FK chain
2. Adapt the RNEA forward/backward recursion to use `LinkSpec` data
3. Use the Addverb's gravity vector `g = [0, 0, -9.81]` (vs UR5e which has a different base frame)
4. Handle the Addverb's unusual joint5 axis `[0.5, 0.866, 0]` (tilted joint, unlike UR5e)

**Key difference from paper's code**: The paper uses `iquat` (inertia frame quaternion) from MuJoCo's body data. For the Addverb URDF, the inertia is already expressed in the link frame (no additional rotation), so we use `inertia_matrix` directly.

```python
class RNEASolver:
    """Recursive Newton-Euler Algorithm for the Addverb cobot."""
    
    def __init__(self, gravity: np.ndarray = np.array([0.0, 0.0, -9.81])):
        self.gravity = gravity

    def compute_mcg(self, q: np.ndarray, dq: np.ndarray, robot: RobotModel):
        """
        Returns:
            M: (n, n) mass matrix
            CG: (n, 1) Coriolis + gravity vector (C(q,dq)*dq + G(q))
            G: (n, 1) gravity vector only
        """
        # G = RNEA(q, dq=0, ddq=0)
        G = self._rnea(q, np.zeros_like(q), np.zeros_like(q), robot)
        
        # CG = RNEA(q, dq, ddq=0)  →  C(q,dq)*dq + G(q)
        CG = self._rnea(q, dq, np.zeros_like(q), robot)
        
        # M via TMT method (J_com^T * M_spatial * J_com for all links)
        M = self._compute_mass_matrix_tmt(q, robot)
        
        return M, CG[:, np.newaxis], G[:, np.newaxis]
    
    def _rnea(self, q, dq, ddq, robot):
        """Full RNEA forward+backward pass, returns n-dim tau vector."""
        # ... (port from paper_code/script/recursive_NE.py, adapting to LinkSpec)
        pass
    
    def _compute_mass_matrix_tmt(self, q, robot):
        """Mass matrix via Transport Matrix (J_g^T * M_spatial * J_g)."""
        # ... (port from paper_code/script/TMT.py, adapting to LinkSpec)
        pass
```

### Step 1.4: Rewrite `ReducedDynamics` to Use RNEA

Port `paper_code/script/reduced_dynamics.py` structure:

```python
class ReducedDynamics:
    def __init__(self, model: DualArmModel):
        self.model = model
        self.rnea_active = RNEASolver()
        self.rnea_passive = RNEASolver()

    def s_matrix(self, q_active, q_passive):
        ja = jacobian(q_active, self.model.right)
        jp = jacobian(q_passive, self.model.left)
        return np.linalg.pinv(ja) @ jp  # Paper uses plain pinv, not damped

    def ds_matrix(self, q_active, q_passive, dq_active, dq_passive, eps=1e-6):
        """Numerical Ṡ — identical to paper_code/script/reduced_dynamics.py line 22-32."""
        s_next = self.s_matrix(q_active + dq_active * eps, q_passive + dq_passive * eps)
        s_prev = self.s_matrix(q_active - dq_active * eps, q_passive - dq_passive * eps)
        return (s_next - s_prev) / (2.0 * eps)

    def reduced_terms(self, q_active, q_passive, dq_active, dq_passive):
        # Get per-arm dynamics via RNEA
        m_active, cg_active, g_active = self.rnea_active.compute_mcg(
            q_active, dq_active, self.model.right)
        m_passive, cg_passive, g_passive = self.rnea_passive.compute_mcg(
            q_passive, dq_passive, self.model.left)

        s = self.s_matrix(q_active, q_passive)
        n = len(q_active)
        m = len(q_passive)

        # N matrix (paper Eq. 8)
        n_matrix = np.vstack((s, np.eye(m)))

        # Mr = N^T M N (paper Eq. 10)
        m_full = np.block([
            [m_active, np.zeros((n, m))],
            [np.zeros((m, n)), m_passive],
        ])
        mr = n_matrix.T @ m_full @ n_matrix

        # CGr = N^T (CG_full + M_full * dN * dq_passive)  (paper Eq. 10, includes Ṅ term)
        dn_matrix = np.vstack((
            self.ds_matrix(q_active, q_passive, dq_active, dq_passive),
            np.zeros((m, m))
        ))
        cg_full = np.vstack((cg_active, cg_passive))
        cgr = n_matrix.T @ (cg_full + m_full @ (dn_matrix @ dq_passive[:, np.newaxis]))

        # Gr = S^T G_active + G_passive (paper Eq. 10)
        gr = s.T @ g_active + g_passive

        return mr, cgr, gr, s
```

**Critical note**: In the paper's controller, `cgr` already includes gravity (since RNEA's CG = C*dq + G). So:
- `u = Mr @ acc_cmd + cgr` is the complete control law (Eq. 21)
- The acceleration equation is `ddq_II = Mr^{-1}(S^T τ_I - cgr)` which also uses `cgr` (C*dq+G combined)
- There is **no separate `hr`** in the paper's actual code — it's absorbed into `cgr`

### Step 1.5: Update `ImpedanceController.compute_tau()` 

Match the paper's `controller.py` exactly:

```python
def compute_tau(self, q_passive, dq_passive, reference, mr, cgr, s):
    e = (q_passive - reference.qd_passive)[:, np.newaxis]
    ed = (dq_passive - reference.dqd_passive)[:, np.newaxis]
    
    acc_cmd = reference.ddqd_passive[:, np.newaxis] - self.m_imp_inv @ (
        self.gains.k_imp @ e + self.gains.d_imp @ ed
    )
    u = mr @ acc_cmd + cgr   # cgr = C_r*dq + G_r (combined, per paper)
    
    tau = damped_pinv(s.T, 1e-2) @ u
    return tau.reshape(-1)
```

**This is actually the current signature!** The current code has the right structure — `cgr` just needs to contain the real C_r*dq+G_r instead of the heuristic.

### Step 1.6: Update `GeometricIntegrator.step()` Acceleration Equation

Current code (line 286):
```python
rhs = s.T @ tau_active[:, None] - cgr - hr  # WRONG: separate cgr and hr
```

Paper's code (`integrator.py` line 94):
```python
ddq_passive = np.linalg.solve(mr, reduced_force - cgr)  # cgr already has gravity
```

Fix:
```python
rhs = s.T @ tau_active[:, None] - cgr  # cgr = C_r*dq + G_r (combined)
```

### Testing
- Compare `Mr`, `CGr`, `Gr` outputs at several known configurations against the paper's UR5e outputs (using similar joint angles). They won't be identical (different robot) but should be similar in structure/magnitude.
- Verify `Mr` is symmetric positive definite at all tested configurations.
- Verify `CG` at zero velocity equals `G` (pure gravity).
- `--dry-run` test: `tau_active` values should be in reasonable range (10–50 Nm for the Addverb's 150 Nm limit).

---

## Phase 2: IK Solver Improvements — `dual_arm_math.py`

**Issues Resolved**: #5 (joint limits), IK robustness

### Step 2.1: Add Joint Limit Clamping

```python
def solve_active_ik(q_seed, target_pos, target_rot, robot, max_iters=100, tol=2e-3, damping=2e-2):
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
        
        # Clamp to joint limits from URDF
        for i, link in enumerate(robot.links):
            q[i] = np.clip(q[i], link.q_min, link.q_max)
    
    fk = forward_kinematics(q, robot)
    res = np.concatenate((fk.ee_pos - target_pos, orientation_error(fk.ee_rot, target_rot)))
    return q, False, float(np.linalg.norm(res))
```

**Note on IK sign convention**: After reviewing the paper's original code (`integrator.py` line 53):
```python
pos_err = fk_candidate.end_eff_pos - target_pos  # current - target
```
Then `least_squares` minimizes the residual → drives pos_err toward 0. The current code's convention (`fk - target` with `dq = -J†*err`) is mathematically equivalent and correct. **Do NOT change the signs** — the current code uses the standard Newton-Raphson convention: `err = f(x)`, `x_new = x - J†*f(x)`.

### Step 2.2: Add NaN/Inf Guard (from paper's integrator.py lines 109-116)

```python
# In GeometricIntegrator.step(), before calling IK:
if (np.any(~np.isfinite(state.q_active)) or 
    np.any(~np.isfinite(q_passive_new)) or 
    np.any(~np.isfinite(dq_passive_new))):
    q_active_new = self.last_valid_q_active.copy() if self.last_valid_q_active is not None else state.q_active.copy()
    ik_ok = False
    ik_res = float("inf")
else:
    q_active_new, ik_ok, ik_res = solve_active_ik(...)
```

### Testing
- Verify IK solutions stay within URDF joint limits for all trajectory points.
- Test with configurations near joint limits — solutions should clamp gracefully.

---

## Phase 3: S Matrix — Match Paper's Convention — `dual_arm_math.py`

### Step 3.1: Use `np.linalg.pinv` Instead of `damped_pinv` for S Matrix

The paper's `reduced_dynamics.py` line 19 uses plain pseudoinverse:
```python
j_active_inv = np.linalg.pinv(j_active)
return j_active_inv @ j_passive
```

Current code uses `damped_pinv(ja, 1e-2)`. While damped is safer near singularities, it introduces a systematic bias. Use plain pinv for S, matching the paper, and add singularity detection as a diagnostic:

```python
def s_matrix(self, q_active, q_passive):
    ja = jacobian(q_active, self.model.right)
    jp = jacobian(q_passive, self.model.left)
    # Use plain pinv to match paper (reduced_dynamics.py line 19)
    w = np.sqrt(max(np.linalg.det(ja @ ja.T), 0.0))  # manipulability
    if w < 1e-6:
        return damped_pinv(ja, 1e-2) @ jp  # fallback near singularity
    return np.linalg.pinv(ja) @ jp
```

### Step 3.2: Recompute S at New Configuration for Active Velocity Update

```python
# In GeometricIntegrator.step(), after IK:
s_new = self.reduced_dynamics.s_matrix(q_active_new, q_passive_new)
dq_active_new = (s_new @ dq_passive_new[:, np.newaxis]).reshape(-1)
```

### Testing
- Compare S matrix values against paper's implementation at same joint angles.
- Log manipulability index `w` as a diagnostic.

---

## Phase 4: ROS2 Interface Refinements — `dual_arm_geometric_control.py`

### Step 4.1: Fuse Sensor Feedback (Close the Loop)

The paper's MuJoCo simulation is inherently closed-loop (MuJoCo handles physics). Our ROS2 code runs open-loop after init. Fix:

```python
# At top of each loop iteration, after spinning:
rclpy.spin_once(node, timeout_sec=0.0)

q_passive_meas = node.get_q(node.left_joint_map)
q_active_meas = node.get_q(node.right_joint_map)

if q_passive_meas is not None and q_active_meas is not None:
    # Blend integrated state with measurements (complementary filter)
    alpha = cfg.get("runtime", {}).get("sensor_blend_alpha", 0.3)
    state = ClosedLoopState(
        q_active=(1 - alpha) * state.q_active + alpha * q_active_meas,
        q_passive=(1 - alpha) * state.q_passive + alpha * q_passive_meas,
        dq_active=state.dq_active,
        dq_passive=state.dq_passive,
    )
```

Add `sensor_blend_alpha: 0.3` to the YAML `runtime` section.

### Step 4.2: Use Measured Position as Trajectory Start

```python
# Before the loop:
q_passive_start_actual = q_passive_0.copy()  # from sensor measurement

# In the loop:
qd, dqd, ddqd = quintic_trajectory(
    t, t0, tf,
    q_passive_start_actual,  # measured, not from YAML
    np.array(task["q_passive_goal"], dtype=float),
)
```

### Step 4.3: Unify dt and rate_hz

```python
dt = float(task["dt"])
# Override rate to match dt:
loop_period = dt  # NOT 1/rate_hz — use integration step as the loop period

# In the loop timing:
loop_start = time.time()
# ... computation + send commands ...
elapsed = time.time() - loop_start
time.sleep(max(0.0, loop_period - elapsed))
```

### Step 4.4: Validate Parameter Setting Success

```python
def set_right_impedance_params(self, diag_k, diag_d, diag_m) -> bool:
    client = AsyncParameterClient(self, "/ar2/cartesian_impedance_controller")
    if not client.wait_for_services(timeout_sec=5.0):
        self.get_logger().error("Parameter services unreachable")
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
        self.get_logger().error("Parameter set failed or timed out")
        return False
    
    for r in fut.result().results:
        if not r.successful:
            self.get_logger().error(f"Parameter rejected: {r.reason}")
            return False
    return True
```

Update call site to check return value and abort if failed.

### Step 4.5: Remove Unnecessary `validate_env_imports()`

`addverb_cobot_msgs` is never actually imported or used in any of the script files. Remove:

```python
# DELETE these lines (205-211):
def validate_env_imports():
    try:
        import addverb_cobot_msgs.msg
    except Exception as exc:
        raise RuntimeError(...)

# DELETE this call (line 215):
validate_env_imports()
```

### Step 4.6: Document `gravity_comp_effort_controller` Semantics

After reading the actual C++ source, the controller is a **raw effort pass-through**. Add documentation:

```python
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
```

### Testing
- Verify sensor feedback reduces constraint drift compared to open-loop.
- Confirm parameter setting success is logged properly.
- Test timing consistency with unified dt.

---

## Phase 5: Model Completeness — `dual_arm_addverb_model.py`

### Step 5.1: Add Base Rotation Support

The two arms may face each other (180° rotation). Currently only translation is modeled.

```python
@dataclass
class RobotModel:
    links: List[LinkSpec]
    base_xyz: np.ndarray
    base_rot: np.ndarray  # 3x3 rotation matrix, default=np.eye(3)
    ee_offset_xyz: np.ndarray

def build_dual_arm_model(left_base_xyz, right_base_xyz, ee_offset_xyz,
                         left_base_rpy=None, right_base_rpy=None):
    # ... Convert RPY to rotation matrix if provided ...
```

Update FK to apply base rotation:
```python
def forward_kinematics(q, robot):
    t = np.eye(4)
    t[:3, 3] = robot.base_xyz
    t[:3, :3] = robot.base_rot  # Apply base rotation
    # ... rest unchanged ...
```

### Step 5.2: Set Correct EE Offset

From URDF, joint6 origin is at `[0, 0, -0.0975]` from link5, and the end-effector frame is the child of joint6. If there's a tool/gripper, its offset should be configured:

```yaml
robot:
  ee_offset_xyz: [0.0, 0.0, 0.0]  # Measure actual tool offset, if any
```

### Step 5.3: Handle Joint 6 Zero Mass

Joint6's child link (`end-effector`) has no inertial in the URDF — it's a virtual frame. For RNEA, we need a small positive mass to avoid singularity:

```python
LinkSpec(
    joint_name="joint6",
    joint_origin_xyz=np.array([0.0, 0.0, -0.0975]),
    joint_axis=np.array([0.0, 0.0, -1.0]),
    com_xyz=np.array([0.0, 0.0, -0.01]),  # Small offset 
    mass=0.1,  # Minimal mass for numerical stability
    inertia_matrix=np.diag([1e-4, 1e-4, 1e-4]),  # Small
    q_min=-np.pi,
    q_max=np.pi,
),
```

### Testing
- Verify FK with base rotation produces correct world-frame positions.
- Verify RNEA handles joint6 without numerical issues.

---

## Phase 6: Diagnostics & Safety — `dual_arm_geometric_control.py` + `dual_arm_math.py`

### Step 6.1: Add Manipulability Diagnostic

```python
# In GeometricIntegrator.step(), compute and log:
ja = jacobian(state.q_active, self.model.right)
manipulability = np.sqrt(max(np.linalg.det(ja @ ja.T), 0.0))
diag["manipulability"] = float(manipulability)
```

### Step 6.2: Add `jcn_norm` Diagnostic (from paper)

The paper's integrator computes `jcn_norm = ||Jc * N||` as a constraint consistency check:

```python
# From paper_code/script/integrator.py lines 134-150:
ja = jacobian(state.q_active, self.model.right)
jp = jacobian(q_passive_new, self.model.left)
n_matrix = np.vstack((s, np.eye(6)))
jc_matrix = np.hstack((ja, -jp))
diag["jcn_norm"] = float(np.linalg.norm(jc_matrix @ n_matrix))
```

### Step 6.3: Add Safety Stops

```python
# In the control loop:
if out.diagnostics["constraint_pos_error"] > 0.1:  # 10cm
    node.get_logger().error("SAFETY: Constraint violation > 10cm, stopping!")
    break

if out.diagnostics["manipulability"] < 1e-4:
    node.get_logger().warn("Near singularity, reducing speed")
```

### Testing
- Verify all new diagnostics appear in CSV output.
- Test safety stop triggers correctly.

---

## Updated YAML Config

```yaml
robot:
  left_base_xyz: [-0.1923, 0.0, 0.0]
  left_base_rpy: [0.0, 0.0, 0.0]
  right_base_xyz: [0.1923, 0.0, 0.0]
  right_base_rpy: [0.0, 0.0, 3.14159]     # If arms face each other
  ee_offset_xyz: [0.0, 0.0, 0.0]

runtime:
  rate_hz: 50.0
  warmup_timeout_sec: 10.0
  output_csv: "cobot_ros2/script/logs/dual_arm_geometric_log.csv"
  sensor_blend_alpha: 0.3

limits:
  tau_max: 120.0
  dq_max: 2.5
  ddq_max: 8.0

tasks:
  push:
    q_passive_start: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    q_passive_goal: [0.2, -0.2, 0.15, 0.0, 0.1, 0.0]
    k_imp_diag: [60.0, 60.0, 60.0, 25.0, 25.0, 25.0]
    d_imp_diag: [8.0, 8.0, 8.0, 5.0, 5.0, 5.0]
    m_imp_diag: [0.08, 0.08, 0.08, 0.05, 0.05, 0.05]
    t0: 0.0
    tf: 6.0
    dt: 0.02                  # Aligned with rate_hz=50

  pull:
    q_passive_start: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    q_passive_goal: [-0.2, 0.18, -0.12, 0.0, -0.08, 0.0]
    k_imp_diag: [75.0, 75.0, 75.0, 30.0, 30.0, 30.0]
    d_imp_diag: [10.0, 10.0, 10.0, 6.0, 6.0, 6.0]
    m_imp_diag: [0.09, 0.09, 0.09, 0.06, 0.06, 0.06]
    t0: 0.0
    tf: 4.0
    dt: 0.02                  # Aligned with rate_hz=50
```

---

## Execution Order & Dependencies

```
Phase 1 (RNEA Dynamics Port)  ←── THE critical fix
    ├── Step 1.1: Full inertia in LinkSpec (from URDF)
    ├── Step 1.2: Joint limits in LinkSpec (from URDF)
    ├── Step 1.3: Port RNEA + TMT to dual_arm_math.py
    ├── Step 1.4: Rewrite ReducedDynamics with proper Mr, CGr, Gr, dS
    ├── Step 1.5: Verify ImpedanceController.compute_tau matches paper
    └── Step 1.6: Fix acceleration equation (use combined cgr, remove separate hr)
         │
Phase 2 (IK Improvements)
    ├── Step 2.1: Joint limit clamping
    └── Step 2.2: NaN/Inf guard
         │ 
Phase 3 (S Matrix Match)
    ├── Step 3.1: Use plain pinv with singularity fallback
    └── Step 3.2: Recompute S at new config
         │
Phase 4 (ROS2 Interface)
    ├── Step 4.1: Sensor feedback fusion
    ├── Step 4.2: Measured trajectory start
    ├── Step 4.3: Unify dt/rate_hz
    ├── Step 4.4: Validate parameter setting
    ├── Step 4.5: Remove unused validate_env_imports
    └── Step 4.6: Document gravity_comp semantics
         │
Phase 5 (Model Completeness)
    ├── Step 5.1: Base rotation
    ├── Step 5.2: EE offset
    └── Step 5.3: Joint 6 mass
         │
Phase 6 (Diagnostics & Safety)
    ├── Step 6.1: Manipulability diagnostic
    ├── Step 6.2: jcn_norm diagnostic
    └── Step 6.3: Safety stops
```

---

## Cross-Reference Verification Summary

| Aspect | Paper Reference Code | Current ROS2 Code | Plan Fix |
|--------|---------------------|-------------------|----------|
| Mass matrix M(q) | TMT.py — full J^T M_spatial J | Constant diagonal | Phase 1.3 — Port TMT |
| Coriolis+Gravity CG | RNEA — forward/backward recursion | `viscous*dq + scale*sin(q)` | Phase 1.3 — Port RNEA |
| Reduced Mr | `N^T M_full N` | `S^T diag @ S + diag` | Phase 1.4 — Full N^T M N |
| Reduced CGr (with Ṅ) | `N^T (CG + M*dN*dq)` | `viscous * dq` | Phase 1.4 — Include dN term |
| Reduced Gr | `S^T G_active + G_passive` | `scale * sin(q_passive)` | Phase 1.4 — Both arms |
| Control law u | `Mr @ acc + cgr` (cgr has G) | `mr @ acc + cgr` (cgr is wrong) | Phase 1.5 — Correct once cgr is right |
| Acceleration ddq | `solve(mr, S^T τ - cgr)` | `solve(mr, S^T τ - cgr - hr)` | Phase 1.6 — Remove separate hr |
| S matrix | `pinv(Ja) @ Jp` | `damped_pinv(Ja) @ Jp` | Phase 3.1 — Plain pinv with fallback |
| Ṡ (dS/dt) | Numerical central diff | Not computed | Phase 1.4 — Port ds_matrix |
| IK solver | `scipy.least_squares` | Manual Newton-Raphson | Phase 2 — Add limits, NaN guard |
| Sensor fusion | N/A (MuJoCo is ground truth) | Open-loop after init | Phase 4.1 — Complementary filter |
| Joint limits | Not in paper | Not in code | Phase 2.1 — From URDF |
| Gravity comp semantics | τ_II = 0 (simulation) | Effort pass-through (hardware) | Phase 4.6 — Document |
| URDF full inertia | MuJoCo model data | `inertia_diag` only | Phase 1.1 — Full 3×3 tensor |
