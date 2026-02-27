#!/usr/bin/env python3
"""Unit tests verifying RNEA, TMT, and reduced dynamics correctness.

Run:  python3 cobot_ros2/script/test_dynamics.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from dual_arm_addverb_model import build_dual_arm_model
from dual_arm_math import (
    ClosedLoopState,
    FKResult,
    GeometricIntegrator,
    ImpedanceController,
    ImpedanceGains,
    IntegratorLimits,
    RNEASolver,
    ReducedDynamics,
    TrajectoryReference,
    forward_kinematics,
    jacobian,
    mass_matrix_tmt,
    quintic_trajectory,
    solve_active_ik,
)


def _make_model():
    return build_dual_arm_model(
        left_base_xyz=[-0.1923, 0.0, 0.0],
        right_base_xyz=[0.1923, 0.0, 0.0],
        ee_offset_xyz=[0.0, 0.0, 0.0],
    )


def test_fk_at_zero():
    """FK at q = 0 should produce a deterministic EE position."""
    model = _make_model()
    q = np.zeros(6)
    fk_l = forward_kinematics(q, model.left)
    fk_r = forward_kinematics(q, model.right)
    print(f"FK left  @ q=0: pos={fk_l.ee_pos}")
    print(f"FK right @ q=0: pos={fk_r.ee_pos}")
    # EE positions should differ only by the base offset (2*0.1923 in x)
    dx = fk_r.ee_pos[0] - fk_l.ee_pos[0]
    print(f"  Δx between arms: {dx:.4f} (expect ≈ {2*0.1923:.4f})")
    assert abs(dx - 2 * 0.1923) < 0.01, f"FK base offset mismatch: {dx}"
    print("PASSED: FK at zero\n")


def test_jacobian_numerical():
    """Compare analytic Jacobian against numerical finite differences."""
    model = _make_model()
    q = np.array([0.1, -0.3, 0.2, 0.0, 0.1, -0.1])
    eps = 1e-6

    j_analytic = jacobian(q, model.right)
    j_numeric = np.zeros_like(j_analytic)

    fk0 = forward_kinematics(q, model.right)
    for i in range(6):
        q_plus = q.copy(); q_plus[i] += eps
        fk_p = forward_kinematics(q_plus, model.right)
        j_numeric[:3, i] = (fk_p.ee_pos - fk0.ee_pos) / eps
        # Skip angular part (harder to compare numerically)

    err = np.linalg.norm(j_analytic[:3, :] - j_numeric[:3, :])
    print(f"Jacobian numeric error (linear part): {err:.2e}")
    assert err < 1e-3, f"Jacobian error too large: {err}"
    print("PASSED: Jacobian numerical check\n")


def test_mass_matrix_properties():
    """M(q) should be symmetric and positive definite for various q."""
    model = _make_model()
    for name, q in [
        ("zero", np.zeros(6)),
        ("random1", np.array([0.2, -0.3, 0.4, 0.1, -0.2, 0.0])),
        ("random2", np.array([-0.5, 0.3, -0.2, 0.8, 0.1, -0.3])),
    ]:
        M = mass_matrix_tmt(q, model.right)
        sym_err = np.linalg.norm(M - M.T)
        eigs = np.linalg.eigvalsh(M)
        min_eig = float(np.min(eigs))
        print(f"M({name}): symmetry_err={sym_err:.2e}, min_eig={min_eig:.6f}")
        assert sym_err < 1e-10, f"M not symmetric: {sym_err}"
        assert min_eig > 0, f"M not positive definite: min_eig={min_eig}"
    print("PASSED: Mass matrix properties\n")


def test_gravity_at_zero_velocity():
    """At dq=0, RNEA should produce CG = G (no Coriolis contribution)."""
    model = _make_model()
    rnea = RNEASolver()
    q = np.array([0.1, -0.3, 0.5, 0.0, 0.1, 0.0])
    dq_zero = np.zeros(6)

    M, CG, G = rnea.compute_mcg(q, dq_zero, model.right)
    err = np.linalg.norm(CG - G)
    print(f"CG - G at dq=0: {err:.2e} (should be ≈ 0)")
    assert err < 1e-10, f"CG ≠ G at zero velocity: {err}"
    print("PASSED: Gravity at zero velocity\n")


def test_gravity_nonzero():
    """G should have nonzero entries for shoulder joints supporting weight."""
    model = _make_model()
    rnea = RNEASolver()
    q = np.zeros(6)
    dq = np.zeros(6)

    M, CG, G = rnea.compute_mcg(q, dq, model.right)
    print(f"G at q=0: {G.flatten()}")
    g_norm = np.linalg.norm(G)
    print(f"  ||G|| = {g_norm:.4f}")
    # For a typical 6-DOF arm with ~13 kg, gravity torques should be several Nm
    assert g_norm > 0.5, f"Gravity vector suspiciously small: {g_norm}"
    print("PASSED: Gravity nonzero check\n")


def test_reduced_dynamics_mr_pd():
    """Reduced mass matrix Mr should be symmetric positive definite."""
    model = _make_model()
    reduced = ReducedDynamics(model)

    qa = np.array([0.1, -0.2, 0.3, 0.0, 0.1, 0.0])
    qp = np.array([-0.1, 0.2, -0.1, 0.0, -0.05, 0.0])
    dqa = np.zeros(6)
    dqp = np.zeros(6)

    mr, cgr, gr, s = reduced.reduced_terms(qa, qp, dqa, dqp)
    sym_err = np.linalg.norm(mr - mr.T)
    eigs = np.linalg.eigvalsh(mr)
    min_eig = float(np.min(eigs))
    print(f"Mr: symmetry_err={sym_err:.2e}, min_eig={min_eig:.6f}")
    print(f"  CGr at dq=0: {cgr.flatten()}")
    print(f"  Gr:          {gr.flatten()}")
    print(f"  S shape:     {s.shape}")
    assert sym_err < 1e-8, f"Mr not symmetric: {sym_err}"
    assert min_eig > 0, f"Mr not positive definite: {min_eig}"
    print("PASSED: Reduced dynamics Mr positive definite\n")


def test_ik_solver():
    """IK should find a joint configuration matching a known FK position."""
    model = _make_model()
    q_true = np.array([0.1, -0.2, 0.3, 0.0, 0.1, 0.0])
    fk = forward_kinematics(q_true, model.right)

    q_seed = q_true + np.random.RandomState(42).randn(6) * 0.05
    q_ik, ok, res = solve_active_ik(q_seed, fk.ee_pos, fk.ee_rot, model.right)
    print(f"IK residual: {res:.6e}, ok: {ok}")
    print(f"  q_true: {q_true}")
    print(f"  q_ik:   {q_ik}")
    assert ok, f"IK failed with residual={res}"
    assert res < 5e-3, f"IK residual too large: {res}"

    # Verify joint limits respected
    for i, link in enumerate(model.right.links):
        assert link.q_min <= q_ik[i] <= link.q_max, \
            f"Joint {i} out of limits: {q_ik[i]} ∉ [{link.q_min}, {link.q_max}]"
    print("PASSED: IK solver\n")


def test_quintic_trajectory():
    """Quintic trajectory should start/end exactly at p0/pf with zero velocity."""
    p0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pf = np.array([0.2, -0.2, 0.15, 0.0, 0.1, 0.0])

    q, dq, ddq = quintic_trajectory(0.0, 0.0, 5.0, p0, pf)
    assert np.allclose(q, p0), f"Start position mismatch: {q}"
    assert np.allclose(dq, 0), f"Start velocity not zero: {dq}"

    q, dq, ddq = quintic_trajectory(5.0, 0.0, 5.0, p0, pf)
    assert np.allclose(q, pf), f"End position mismatch: {q}"
    assert np.allclose(dq, 0), f"End velocity not zero: {dq}"

    q, dq, ddq = quintic_trajectory(2.5, 0.0, 5.0, p0, pf)
    print(f"Mid-point: q={q}")
    assert np.allclose(q, 0.5 * (p0 + pf), atol=0.05), "Mid trajectory not near midpoint"
    print("PASSED: Quintic trajectory\n")


def test_integrator_single_step():
    """Run one integration step and verify finite outputs."""
    model = _make_model()
    reduced = ReducedDynamics(model)
    gains = ImpedanceGains(
        k_imp=np.diag([60.0] * 3 + [25.0] * 3),
        d_imp=np.diag([8.0] * 3 + [5.0] * 3),
        m_imp=np.diag([0.08] * 3 + [0.05] * 3),
    )
    controller = ImpedanceController(gains)
    limits = IntegratorLimits(tau_max=120.0, dq_max=2.5, ddq_max=8.0)
    integrator = GeometricIntegrator(model, reduced, controller, limits)

    state = ClosedLoopState(
        q_active=np.zeros(6),
        q_passive=np.zeros(6),
        dq_active=np.zeros(6),
        dq_passive=np.zeros(6),
    )

    fk_a = forward_kinematics(state.q_active, model.right)
    fk_p = forward_kinematics(state.q_passive, model.left)
    r_rel = fk_p.ee_rot.T @ fk_a.ee_rot

    ref = TrajectoryReference(
        qd_passive=np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
        dqd_passive=np.zeros(6),
        ddqd_passive=np.zeros(6),
    )

    out = integrator.step(state, ref, 0.02, r_rel)
    print(f"Step output diagnostics:")
    for k, v in out.diagnostics.items():
        print(f"  {k}: {v}")

    assert np.all(np.isfinite(out.state.q_active)), "q_active not finite"
    assert np.all(np.isfinite(out.state.q_passive)), "q_passive not finite"
    assert np.all(np.isfinite(out.tau_active)), "tau_active not finite"
    print("PASSED: Integrator single step\n")


if __name__ == "__main__":
    tests = [
        test_fk_at_zero,
        test_jacobian_numerical,
        test_mass_matrix_properties,
        test_gravity_at_zero_velocity,
        test_gravity_nonzero,
        test_reduced_dynamics_mr_pd,
        test_ik_solver,
        test_quintic_trajectory,
        test_integrator_single_step,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            print(f"--- {test.__name__} ---")
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed > 0:
        sys.exit(1)
