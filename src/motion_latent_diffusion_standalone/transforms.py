"""
Motion Transformation Functions

Utilities for converting between motion feature representations and 3D joint positions.
These functions implement the RIC (Rotation-Invariant Coordinates) representation from
HumanML3D, which encodes motion relative to the root joint for better generalization.

Key concepts:
- RIC features: Root velocity + root height + local joint positions (rotation-invariant)
- Quaternions: Used for representing rotations (4D: w, x, y, z)
- Root recovery: Integrate velocities to get absolute position/rotation over time

Extracted from mld/data/humanml/scripts/motion_process.py
"""

import torch


def qinv(q: torch.Tensor) -> torch.Tensor:
    """
    Compute quaternion inverse (conjugate for unit quaternions).

    For a quaternion q = (w, x, y, z), the inverse is q* = (w, -x, -y, -z).
    This effectively reverses the rotation represented by the quaternion.

    Args:
        q: Quaternions of shape (..., 4) in (w, x, y, z) format

    Returns:
        Inverted quaternions (..., 4)
    """
    assert q.shape[-1] == 4, "q must be a tensor of shape (*, 4)"
    # Create mask: [1, -1, -1, -1] to negate imaginary components
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qrot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector(s) v by quaternion(s) q.

    Applies 3D rotation using quaternion multiplication formula:
    v' = v + 2 * w * (u × v) + 2 * (u × (u × v))
    where q = (w, u) with u = (x, y, z) being the imaginary part.

    Args:
        q: Quaternions (..., 4) in (w, x, y, z) format
        v: 3D vectors (..., 3)

    Returns:
        Rotated vectors (..., 3)
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    # Extract imaginary part (rotation axis * sin(θ/2))
    qvec = q[..., 1:]

    # Apply quaternion rotation formula using cross products
    # uv = u × v
    uv = torch.cross(qvec, v, dim=-1)
    # uuv = u × (u × v)
    uuv = torch.cross(qvec, uv, dim=-1)
    # v' = v + 2w(u × v) + 2(u × (u × v))
    return v + 2 * (q[..., :1] * uv + uuv)


def recover_root_rot_pos(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Recover root joint rotation and position by integrating velocities.

    HumanML3D stores root motion as velocities (rotation and XZ translation) rather
    than absolute pose. This function integrates these velocities over time to recover
    the absolute root trajectory. The root rotates around Y-axis (yaw) only.

    Feature format (263-dim for HumanML3D):
    - [0]: root rotation velocity (angular velocity around Y)
    - [1:3]: root XZ linear velocity (in local frame)
    - [3]: root Y position (height, absolute)
    - [4:]: local joint positions, rotations, velocities, contacts

    Args:
        data: Motion features (..., 263) for HumanML3D

    Returns:
        r_rot_quat: Root rotation quaternions (..., 4) - rotation around Y-axis
        r_pos: Root 3D positions (..., 3) - absolute world coordinates
    """
    # Extract root rotation velocity (angular velocity around Y-axis)
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros(data.shape[:-1] + (1,)).to(data.device)

    # Integrate rotation velocity: θ(t) = Σ ω(t) * dt
    # Note: velocities at frame t give change to frame t+1
    r_rot_ang[..., 1:, :] = rot_vel[..., :-1, None]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-2)

    # Convert rotation angle to quaternion (Y-axis rotation only)
    # Y-axis quaternion: q = (cos(θ/2), 0, sin(θ/2), 0)
    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang[..., 0])  # w component
    r_rot_quat[..., 2] = torch.sin(r_rot_ang[..., 0])  # y component

    # Recover root XZ position from velocity
    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]  # XZ velocities (in local frame)

    # Transform local velocities to world frame using inverse root rotation
    # This accounts for the character's changing orientation
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    # Integrate to get absolute position: p(t) = Σ v(t) * dt
    r_pos = torch.cumsum(r_pos, dim=-2)

    # Set Y-axis (height) from absolute value stored in features
    r_pos[..., 1] = data[..., 3]

    return r_rot_quat, r_pos


def recover_from_ric(data: torch.Tensor, joints_num: int) -> torch.Tensor:
    """
    Convert RIC (Rotation-Invariant Coordinates) to 3D joint positions.

    RIC representation stores joint positions relative to the root orientation,
    making the features invariant to global rotation. This function:
    1. Recovers root rotation and position by integrating velocities
    2. Extracts local joint positions (stored in root's reference frame)
    3. Applies inverse root rotation to transform to world coordinates
    4. Translates by root position to get absolute joint locations

    This representation is beneficial because:
    - Makes features rotation-invariant (better for learning)
    - Separates root motion from body pose
    - Allows flexible trajectory editing

    Args:
        data: Motion features (..., 263 for HumanML3D or 251 for KIT)
        joints_num: Number of joints (22 for HumanML3D, 21 for KIT)

    Returns:
        positions: Absolute 3D joint positions (..., joints_num, 3)
    """
    # Step 1: Recover root trajectory (rotation and position)
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    # Step 2: Extract local joint positions from features
    # Features [4 : 4+(joints_num-1)*3] contain XYZ for each non-root joint
    # These are stored in the root's local coordinate frame
    positions = data[..., 4 : (joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))  # Reshape to (*, J-1, 3)

    # Step 3: Transform from root-local to world coordinates
    # Apply inverse root rotation to align with world frame
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(
            positions.shape[:-1] + (4,)
        ),  # Broadcast quaternion
        positions,
    )

    # Step 4: Translate by root position
    # Add root X and Z (horizontal plane) - Y is already in world coords in local positions
    positions[..., 0] += r_pos[..., 0:1]  # Add root X
    positions[..., 2] += r_pos[..., 2:3]  # Add root Z

    # Step 5: Prepend root joint to create full skeleton
    # Result: [root, joint1, joint2, ..., jointN]
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


class MotionTransform:
    """
    Helper class for motion transformations.
    Mimics the datamodule.feats2joints interface used in the full model.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        njoints: int = 22,
    ) -> None:
        """
        Args:
            mean: Mean normalization values (numpy array or torch tensor)
            std: Std normalization values (numpy array or torch tensor)
            njoints: Number of joints
        """
        self.mean = mean
        self.std = std
        self.njoints = njoints

    def feats2joints(self, features: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized features to 3D joint positions.

        Args:
            features: Normalized motion features (batch, seq_len, nfeats)

        Returns:
            joints: 3D joint positions (batch, seq_len, njoints, 3)
        """
        # Denormalize using precomputed mean and std
        mean = self.mean.to(features.device)
        std = self.std.to(features.device)
        features = features * std + mean

        # Convert from RIC representation to 3D joints
        joints = recover_from_ric(features, self.njoints)

        return joints
