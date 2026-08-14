"""Pin Apo2Mol's inlined ``kornia`` conversions.

The integration deliberately does not depend on ``kornia``; three of its
conversions are reimplemented in
``modules/models/apo2mol/residue_ops.py``. The released Apo2Mol checkpoint was
trained under kornia's exact conventions -- ``(w, x, y, z)`` quaternions and
``R @ v`` on column vectors -- and a sign or ordering slip would be **silent**:
the model would still emit plausible pockets, just wrong ones. Hence hard
values, not just round trips.
"""

from __future__ import annotations

import math

import pytest
import torch

from MolecularDiffusion.modules.models.apo2mol.residue_ops import (
    axis_angle_to_quaternion,
    axis_angle_to_rotation_matrix,
    quaternion_product,
    quaternion_to_rotation_matrix,
    slerp_identity_to_q,
)

#: 90 degrees about +z as a (w, x, y, z) unit quaternion.
_S = 1.0 / math.sqrt(2.0)
Q_Z90 = torch.tensor([[_S, 0.0, 0.0, _S]])
AA_Z90 = torch.tensor([[0.0, 0.0, math.pi / 2]])
R_Z90 = torch.tensor(
    [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
)


def test_quaternion_to_rotation_matrix_hardcoded():
    got = quaternion_to_rotation_matrix(Q_Z90)
    assert torch.allclose(got, R_Z90, atol=1e-6)


def test_rotation_acts_on_column_vectors():
    """``R @ v``, not ``v @ R``: x-hat must go to y-hat, not to -y-hat."""
    r = quaternion_to_rotation_matrix(Q_Z90)[0]
    x_hat = torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(r @ x_hat, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)


def test_axis_angle_conversions_hardcoded():
    assert torch.allclose(axis_angle_to_rotation_matrix(AA_Z90), R_Z90, atol=1e-6)
    assert torch.allclose(axis_angle_to_quaternion(AA_Z90), Q_Z90, atol=1e-6)


def test_quaternion_is_w_first():
    """A rotation about +z must put its sine in the LAST slot, not the second."""
    q = axis_angle_to_quaternion(AA_Z90)[0]
    assert q[0] == pytest.approx(math.cos(math.pi / 4), abs=1e-6)
    assert q[1] == pytest.approx(0.0, abs=1e-6)
    assert q[3] == pytest.approx(math.sin(math.pi / 4), abs=1e-6)


def test_zero_rotation_is_finite():
    """kornia's Taylor branch: theta = 0 must not divide by zero."""
    zero = torch.zeros(1, 3)
    assert torch.allclose(
        axis_angle_to_rotation_matrix(zero), torch.eye(3)[None], atol=1e-6
    )
    assert torch.allclose(
        axis_angle_to_quaternion(zero), torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    )


def test_conversions_agree_and_stay_orthonormal():
    torch.manual_seed(0)
    axis_angle = torch.randn(64, 3)
    quat = axis_angle_to_quaternion(axis_angle)
    rot = quaternion_to_rotation_matrix(quat)

    assert torch.allclose(rot, axis_angle_to_rotation_matrix(axis_angle), atol=1e-5)
    eye = torch.eye(3).expand_as(rot)
    assert torch.allclose(rot @ rot.transpose(-1, -2), eye, atol=1e-5)
    # det +1, i.e. a rotation and not a reflection
    assert torch.allclose(torch.linalg.det(rot), torch.ones(64), atol=1e-5)


def test_quaternion_product_is_hamilton():
    """Two 90-degree z rotations compose to 180 degrees about z."""
    assert torch.allclose(
        quaternion_product(Q_Z90, Q_Z90),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        atol=1e-6,
    )


def test_slerp_endpoints():
    """``lambdas`` weights the IDENTITY end: 1 -> identity, 0 -> q."""
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(
        slerp_identity_to_q(Q_Z90, torch.ones(1, 1)), identity, atol=1e-6
    )
    assert torch.allclose(
        slerp_identity_to_q(Q_Z90, torch.zeros(1, 1)), Q_Z90, atol=1e-6
    )


def test_slerp_midpoint_is_half_the_angle():
    half = slerp_identity_to_q(Q_Z90, torch.full((1, 1), 0.5))
    expected = axis_angle_to_quaternion(torch.tensor([[0.0, 0.0, math.pi / 4]]))
    assert torch.allclose(half, expected, atol=1e-6)
