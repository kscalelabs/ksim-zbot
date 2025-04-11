"""Walking Zbot task with reference gait tracking."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import attrs
import glm
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import xax

try:
    import bvhio
    from bvhio.lib.hierarchy import Joint as BvhioJoint
except ImportError as e:
    raise ImportError(
        "In order to use reference motion utilities, please install Bvhio, using 'pip install bvhio'."
    ) from e


import ksim
from jaxtyping import Array, PRNGKeyArray
from ksim.types import PhysicsModel
from ksim.utils.reference_motion import (
    ReferenceMapping,
    get_local_xpos,
    get_reference_cartesian_poses,
    get_reference_joint_id,
    get_reference_qpos,
    visualize_reference_motion,
)
from ksim.viewer import GlfwMujocoViewer
from scipy.spatial.transform import Rotation as R

from ksim_zbot.zbot2.walking.walking import (
    FeetContactPenalty,
    JointDeviationPenalty,
    LinearVelocityTrackingReward,
    TerminationPenalty,
    ZbotModel,
    ZbotWalkingTask,
    ZbotWalkingTaskConfig,
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MotionAuxOutputs:
    tracked_pos: xax.FrozenDict[int, Array]


@dataclass
class ZbotWalkingReferenceMotionTaskConfig(ZbotWalkingTaskConfig):
    bvh_path: str = xax.field(
        value=str(Path(__file__).parent / "data" / "walk-relaxed_actorcore.bvh"),
        help="The path to the BVH file.",
    )
    rotate_bvh_euler: tuple[float, float, float] = xax.field(
        value=(0, 0, 0),
        help="Optional rotation to ensure the BVH tree matches the Mujoco model.",
    )
    bvh_scaling_factor: float = xax.field(
        value=1.0,
        help="Scaling factor to ensure the BVH tree matches the Mujoco model.",
    )
    reference_motion_offset: tuple[float, float, float] = xax.field(
        value=(0.0, 0.0, 0.0),
        help="Offset to apply to the reference motion (x, y, z).",
    )
    mj_base_name: str = xax.field(
        value="floating_base_link",
        help="The Mujoco body name of the base of the zbot",
    )
    reference_base_name: str = xax.field(
        value="CC_Base_Pelvis",
        help="The BVH joint name of the base of the humanoid",
    )
    visualize_reference_motion: bool = xax.field(
        value=False,
        help="Whether to visualize the reference motion.",
    )
    use_ik_visualization: bool = xax.field(
        value=False,
        help="Whether to use IK to make the robot follow the reference motion during visualization.",
    )
    constrained_joint_ids: tuple[int, ...] = xax.field(
        value=(0, 1, 2, 3, 4, 5, 6),
        help="IDs of joints that should be constrained during IK (typically the floating base).",
    )


# Mapping from BVH joints to Zbot body parts
# This needs to be customized based on your specific BVH file and Zbot model
ZBOT_REFERENCE_MAPPINGS = (
    # Left leg
    ReferenceMapping("CC_Base_L_ThighTwist01", "Left_Hip_Roll_STS3250"),  # hip
    ReferenceMapping("CC_Base_L_CalfTwist01", "Left_Knee_Pitch_STS3250"),  # knee
    ReferenceMapping("CC_Base_L_Foot", "Left_Foot"),  # foot
    # Right leg
    ReferenceMapping("CC_Base_R_ThighTwist01", "Right_Hip_Roll_STS3250"),  # hip
    ReferenceMapping("CC_Base_R_CalfTwist01", "Right_Knee_Pitch_STS3250"),  # knee
    ReferenceMapping("CC_Base_R_Foot", "Right_Foot"),  # foot
    # Left arm
    ReferenceMapping("CC_Base_L_UpperarmTwist01", "Left_Shoulder_Roll_STS3250"),  # shoulder
    ReferenceMapping("CC_Base_L_ForearmTwist01", "R_ARM_MIRROR_1"),  # elbow
    ReferenceMapping("CC_Base_L_Hand", "Left_Finger"),  # hand
    # Right arm
    ReferenceMapping("CC_Base_R_UpperarmTwist01", "Right_Shoulder_Roll_STS3250"),  # shoulder
    ReferenceMapping("CC_Base_R_ForearmTwist01", "L_ARM_MIRROR_1"),  # elbow
    ReferenceMapping("CC_Base_R_Hand", "Right_Finger"),  # hand
)


@attrs.define(frozen=True, kw_only=True)
class ReferenceMotionReward(ksim.Reward):
    reference_motion: xax.FrozenDict[int, xax.HashableArray]
    ctrl_dt: float
    norm: xax.NormType = attrs.field(default="l1")
    sensitivity: float = attrs.field(default=5.0)

    @property
    def num_frames(self) -> int:
        return list(self.reference_motion.values())[0].array.shape[0]

    def __call__(self, trajectory: ksim.Trajectory, reward_carry: None) -> tuple[Array, None]:
        assert isinstance(trajectory.aux_outputs, MotionAuxOutputs)
        reference_motion: xax.FrozenDict[int, Array] = jax.tree.map(lambda x: x.array, self.reference_motion)
        step_number = jnp.int32(jnp.round(trajectory.timestep / self.ctrl_dt)) % self.num_frames
        target_pos = jax.tree.map(lambda x: jnp.take(x, step_number, axis=0), reference_motion)
        tracked_pos = trajectory.aux_outputs.tracked_pos
        error = jax.tree.map(lambda target, tracked: xax.get_norm(target - tracked, self.norm), target_pos, tracked_pos)
        mean_error_over_bodies = jax.tree.reduce(jnp.add, error) / len(error)
        mean_error = mean_error_over_bodies.mean(axis=-1)
        reward = jnp.exp(-mean_error * self.sensitivity)
        return reward, None


class ZbotWalkingReferenceMotionTask(ZbotWalkingTask):
    reference_motion: Optional[xax.FrozenDict[int, xax.HashableArray]] = None
    tracked_body_ids: Optional[tuple[int, ...]] = None
    mj_base_id: Optional[int] = None
    reference_qpos: Optional[np.ndarray] = None

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        if self.reference_motion is None:
            # If we're just setting up the task, return base rewards
            return super().get_rewards(physics_model)

        rewards = [
            ksim.BaseHeightRangeReward(z_lower=0.2, z_upper=0.5, dropoff=10.0, scale=0.5),
            ksim.LinearVelocityPenalty(index="z", scale=-0.01),
            ksim.AngularVelocityPenalty(index="x", scale=-0.01),
            ksim.AngularVelocityPenalty(index="y", scale=-0.01),
            LinearVelocityTrackingReward(scale=0.1),
            ReferenceMotionReward(reference_motion=self.reference_motion, ctrl_dt=self.config.ctrl_dt, scale=0.1),
            JointDeviationPenalty(scale=-0.05),
            TerminationPenalty(scale=-5.0),
            FeetContactPenalty(
                contact_obs_key="contact_observation_feet",
                scale=-0.25,
            ),
        ]

        return rewards

    def sample_action(
        self,
        model: ZbotModel,
        model_carry: None,
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> ksim.Action:
        action_n = super().sample_action(model, model_carry, physics_model, physics_state, observations, commands, rng)

        # Only track positions if we have initialized the reference motion
        if hasattr(self, "tracked_body_ids") and self.tracked_body_ids is not None:
            # Getting the local cartesian positions for all tracked bodies.
            tracked_positions: dict[int, Array] = {}
            for body_id in self.tracked_body_ids:
                body_pos = get_local_xpos(physics_state.data.xpos, body_id, self.mj_base_id)
                tracked_positions[body_id] = jnp.array(body_pos)

            return ksim.Action(
                action=action_n.action,
                aux_outputs=MotionAuxOutputs(
                    tracked_pos=xax.FrozenDict(tracked_positions),
                ),
            )

        return action_n

    def setup_reference_motion(self) -> tuple[PhysicsModel, xax.FrozenDict[int, np.ndarray]]:
        """Set up the reference motion data from BVH file."""
        mj_model: PhysicsModel = self.get_mujoco_model()
        root: BvhioJoint = bvhio.readAsHierarchy(self.config.bvh_path)
        reference_base_id = get_reference_joint_id(root, self.config.reference_base_name)
        self.mj_base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, self.config.mj_base_name)

        def rotation_callback(root: BvhioJoint) -> None:
            euler_rotation = np.array(self.config.rotate_bvh_euler)
            quat = R.from_euler("xyz", euler_rotation).as_quat(scalar_first=True)
            root.applyRotation(glm.quat(*quat), bake=True)

        # Convert offset tuple to numpy array
        offset = np.array(self.config.reference_motion_offset)
        print(f"Applying offset to reference motion: {offset}")

        np_reference_motion = get_reference_cartesian_poses(
            mappings=ZBOT_REFERENCE_MAPPINGS,
            model=mj_model,
            root=root,
            reference_base_id=reference_base_id,
            root_callback=rotation_callback,
            scaling_factor=self.config.bvh_scaling_factor,
            offset=offset,
        )

        # If we're using IK visualization, also compute the reference joint angles
        if self.config.use_ik_visualization:
            print("Computing IK solution for reference motion...")
            try:
                self.reference_qpos = get_reference_qpos(
                    model=mj_model,
                    mj_base_id=self.mj_base_id,
                    bvh_root=root,
                    bvh_to_mujoco_names=ZBOT_REFERENCE_MAPPINGS,
                    bvh_base_id=reference_base_id,
                    bvh_offset=offset,
                    bvh_root_callback=rotation_callback,
                    bvh_scaling_factor=self.config.bvh_scaling_factor,
                    constrained_joint_ids=self.config.constrained_joint_ids,
                    neutral_qpos=None,
                    neutral_similarity_weight=0.1,
                    temporal_consistency_weight=0.1,
                    n_restarts=3,
                    error_acceptance_threshold=1e-4,
                    ftol=1e-8,
                    xtol=1e-8,
                    max_nfev=2000,
                    verbose=True,
                )
                print(f"Successfully computed reference qpos with shape {self.reference_qpos.shape}")
            except Exception as e:
                print(f"Error computing reference qpos: {e}")
                self.reference_qpos = None

        return mj_model, np_reference_motion

    def visualize_reference_motion_with_ik(self, model: mujoco.MjModel) -> None:
        """Visualize the reference motion with the robot following it using precomputed IK."""
        if self.reference_qpos is None:
            print("No reference qpos available. Please ensure IK computation was successful.")
            return

        print("Visualizing reference motion with IK...")
        data = mujoco.MjData(model)

        # Use the GlfwMujocoViewer for better control
        viewer = GlfwMujocoViewer(model, data, mode="window", width=1024, height=768)

        # Set some nice camera parameters
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 45.0
        viewer.cam.elevation = -20.0

        # Total number of frames
        total_frames = len(self.reference_qpos)
        frame = 0

        # The main visualization loop
        while viewer.is_alive:
            frame = frame % total_frames

            # Update the pose using the precomputed reference qpos
            data.qpos = self.reference_qpos[frame]
            mujoco.mj_forward(model, data)

            # Advance time (controls playback speed)
            data.time += model.opt.timestep

            # Render
            viewer.render()

            # Move to next frame
            frame += 1

            # Slow down playback slightly
            time.sleep(0.03)

    def run(self) -> None:
        """Main entry point that handles both visualization and training."""
        # Setup reference motion data
        mj_model, np_reference_motion = self.setup_reference_motion()

        # Convert to JAX types for training
        self.reference_motion = jax.tree.map(lambda x: xax.hashable_array(jnp.array(x)), np_reference_motion)
        self.tracked_body_ids = tuple(self.reference_motion.keys())

        # Decide whether to visualize or train
        if self.config.visualize_reference_motion:
            if self.config.use_ik_visualization and self.reference_qpos is not None:
                self.visualize_reference_motion_with_ik(mj_model)
            else:
                print("Visualizing reference motion markers...")
                visualize_reference_motion(
                    mj_model,
                    base_id=self.mj_base_id,
                    reference_motion=np_reference_motion,
                )
        else:
            print("Starting training...")
            super().run()


if __name__ == "__main__":
    # To run visualization:
    #   python -m ksim_zbot.zbot2.walking.walking_reference_motion visualize_reference_motion=True
    #   python -m ksim_zbot.zbot2.walking.walking_reference_motion use_ik_visualization=True
    # To run training:
    #   python -m ksim_zbot.zbot2.walking.walking_reference_motion visualize_reference_motion=False
    ZbotWalkingReferenceMotionTask.launch(
        ZbotWalkingReferenceMotionTaskConfig(
            num_envs=1024,
            batch_size=256,
            num_passes=10,
            epochs_per_log_step=1,
            valid_every_n_steps=10,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            max_action_latency=0.005,
            min_action_latency=0.0,
            rollout_length_seconds=5.0,
            # PPO parameters
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.001,
            learning_rate=3e-4,
            clip_param=0.3,
            max_grad_norm=1.0,
            # Gait matching parameters.
            bvh_path=str(Path(__file__).parent / "data" / "walk-relaxed_actorcore.bvh"),
            rotate_bvh_euler=(0, np.pi / 2, 0),
            bvh_scaling_factor=1 / 300,  # Scale down significantly for ZBot
            reference_motion_offset=(0.0, 0.0, -0.11),  # Move down by 0.11 units
            mj_base_name="floating_base_link",
            reference_base_name="CC_Base_Pelvis",
            visualize_reference_motion=True,  # Set to True by default for visualization
            use_ik_visualization=True,  # Use IK visualization by default
            # ZBot specific parameters
            action_scale=1.0,
        ),
    )
