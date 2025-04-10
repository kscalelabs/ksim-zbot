"""Defines simple task for training a walking policy for Z-Bot."""

import logging
from dataclasses import dataclass
from typing import Self

import attrs
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import optax
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree
from ksim import Reward
from ksim.curriculum import ConstantCurriculum, Curriculum
from ksim.observation import ContactObservation, ObservationState
from ksim.types import PhysicsState, Trajectory
from ksim.utils.mujoco import get_qpos_data_idxs_by_name
from xax.utils.types.frozen_dict import FrozenDict

from ksim_zbot.zbot2.common import ZbotTask, ZbotTaskConfig

logger = logging.getLogger(__name__)

# Constants for history handling
HISTORY_LENGTH = 0
SINGLE_STEP_HISTORY_SIZE = 0

# Update observation size to match expected model input dimensions
# Previous calculation: 20 joints pos + 20 joints vel + 6 IMU + 2 cmd + 20 last action
# The actual observation might have different sizes based on the logs
# Intended observation components:
# joint_pos_n: 20
# joint_vel_n: 20 (after fixing DHJointVelocityObservation)
# imu_acc_3: 3
# imu_gyro_3: 3
# last_action_n: 20
OBS_SIZE = 20 + 20 + 3 + 3 + 20  # = 66
# Command size:
# lin_vel_cmd_2: 2
CMD_SIZE = 2
NUM_INPUTS = OBS_SIZE + CMD_SIZE
# Final Input Size:
NUM_INPUTS = OBS_SIZE + CMD_SIZE
# NUM_INPUTS = 66 + 2 = 68
NUM_OUTPUTS = 20


@attrs.define(frozen=True)
class HistoryObservation(ksim.Observation):
    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        return jnp.zeros(0, dtype=jnp.float32)


@attrs.define(frozen=True)
class LastActionObservation(ksim.Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        return state.physics_state.most_recent_action

    def add_noise(self, observation: Array, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        return observation + jax.random.normal(rng, observation.shape) * self.noise


class NaiveVelocityReward(ksim.Reward):
    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        return trajectory.qvel[..., 0].clip(max=5.0)


@attrs.define(frozen=True, kw_only=True)
class JointDeviationPenalty(ksim.Reward):
    """Penalty for joint deviations."""

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        # Handle both 1D and 2D arrays
        if trajectory.qpos.ndim > 1:
            diff = trajectory.qpos[:, 7:] - jnp.zeros_like(trajectory.qpos[:, 7:])
            x = jnp.sum(jnp.square(diff), axis=-1)
        else:
            # 1D case for run_environment mode
            diff = trajectory.qpos[7:] - jnp.zeros_like(trajectory.qpos[7:])
            x = jnp.sum(jnp.square(diff))
        return x


@attrs.define(frozen=True, kw_only=True)
class TargetedJointDeviationPenalty(ksim.Reward):
    """Penalty for joint deviations."""

    norm: xax.NormType = attrs.field(default="l2")
    joint_targets: tuple[float, ...] = attrs.field()

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        diff = trajectory.qpos[..., 7:] - jnp.array(self.joint_targets)
        return xax.get_norm(diff, self.norm).sum(axis=-1)


@attrs.define(frozen=True, kw_only=True)
class HeightReward(ksim.Reward):
    """Reward for how high the robot is."""

    height_target: float = attrs.field(default=1.4)

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        if trajectory.qpos.ndim > 1:
            height = trajectory.qpos[:, 2]
        else:
            # 1D case for run_environment mode
            height = trajectory.qpos[2]
        reward = jnp.exp(-jnp.abs(height - self.height_target) * 10)
        return reward


@attrs.define(frozen=True)
class DHJointVelocityObservation(ksim.Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        qvel = state.physics_state.data.qvel[6:]  # (N,)
        return qvel

    def add_noise(self, observation: Array, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        return observation + jax.random.normal(rng, observation.shape) * self.noise


@attrs.define(frozen=True)
class DHJointPositionObservation(ksim.Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        qpos = state.physics_state.data.qpos[7:]  # (N,)
        return qpos

    def add_noise(self, observation: Array, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        return observation + jax.random.normal(rng, observation.shape) * self.noise


@attrs.define(frozen=True, kw_only=True)
class DHForwardReward(ksim.Reward):
    """Incentives forward movement."""

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        # Take just the x velocity component
        if trajectory.qvel.ndim > 1:
            x_delta = -jnp.clip(trajectory.qvel[..., 1], -1.0, 1.0)
        else:
            x_delta = -jnp.clip(trajectory.qvel[1], -1.0, 1.0)
        return x_delta


@attrs.define(frozen=True, kw_only=True)
class DHControlPenalty(ksim.Reward):
    """Legacy default humanoid control cost that penalizes squared action magnitude."""

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        return jnp.sum(jnp.square(trajectory.action), axis=-1)


@attrs.define(frozen=True, kw_only=True)
class DHHealthyReward(ksim.Reward):
    """Legacy default humanoid healthy reward that gives binary reward based on height."""

    healthy_z_lower: float = attrs.field(default=0.2)
    healthy_z_upper: float = attrs.field(default=0.5)

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        height = trajectory.qpos[..., 2]
        is_healthy = jnp.where(height < self.healthy_z_lower, 0.0, 1.0)
        is_healthy = jnp.where(height > self.healthy_z_upper, 0.0, is_healthy)
        return is_healthy


@attrs.define(frozen=True, kw_only=True)
class LinearVelocityTrackingReward(ksim.Reward):
    """Reward for tracking the linear velocity."""

    error_scale: float = attrs.field(default=0.25)
    linvel_obs_name: str = attrs.field(default="sensor_observation_base_link_vel")
    command_name: str = attrs.field(default="linear_velocity_step_command")
    norm: xax.NormType = attrs.field(default="l2")

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        if self.linvel_obs_name not in trajectory.obs:
            raise ValueError(f"Observation {self.linvel_obs_name} not found; add it as an observation in your task.")
        lin_vel_error = xax.get_norm(
            trajectory.command[self.command_name][..., :2] - trajectory.obs[self.linvel_obs_name][..., :2], self.norm
        ).sum(axis=-1)
        return jnp.exp(-lin_vel_error / self.error_scale)


@attrs.define(frozen=True, kw_only=True)
class HipDeviationPenalty(ksim.Reward):
    """Penalty for hip joint deviations."""

    norm: xax.NormType = attrs.field(default="l2")
    hip_indices: tuple[int, ...] = attrs.field()
    joint_targets: tuple[float, ...] = attrs.field()

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        # NOTE - fix that
        diff = (
            trajectory.qpos[..., jnp.array(self.hip_indices) + 7]
            - jnp.array(self.joint_targets)[jnp.array(self.hip_indices)]
        )
        return xax.get_norm(diff, self.norm).sum(axis=-1)

    @classmethod
    def create(
        cls,
        physics_model: ksim.PhysicsModel,
        hip_names: tuple[str, ...],
        joint_targets: tuple[float, ...],
        scale: float = -1.0,
    ) -> Self:
        """Create a sensor observation from a physics model."""
        mappings = get_qpos_data_idxs_by_name(physics_model)
        hip_indices = tuple([int(mappings[name][0]) - 7 for name in hip_names])
        return cls(
            hip_indices=hip_indices,
            joint_targets=joint_targets,
            scale=scale,
        )


@attrs.define(frozen=True, kw_only=True)
class FeetContactPenalty(Reward):
    """Penalizes the robot when foot contact is detected.

    This reward reads the precomputed contact observation from the trajectory. The
    observation is expected to be a boolean array of shape (2,) in a single-environment
    rollout or (num_envs, 2) in batched training mode. The reward reduces along the
    last axis to yield a scalar per environment.
    """

    # The key under which the contact observation is stored.
    contact_obs_key: str = attrs.field(default="contact_observation_feet")
    # Use the reward object's scale field to set the penalty magnitude.
    scale: float = attrs.field(default=-1.0)

    def __call__(self, trajectory: Trajectory) -> jnp.ndarray:
        target_shape = trajectory.done.shape  # Expected shape: () for a single env or (num_envs,) in batched mode.
        contact_flag = trajectory.obs[self.contact_obs_key]

        # If running in batched mode, ensure contact_flag has one extra dimension at the end.
        if target_shape and contact_flag.ndim < (len(target_shape) + 1):
            contact_flag = jnp.broadcast_to(contact_flag, target_shape + (contact_flag.shape[-1],))

        # For single-environment mode (target_shape == ()), simply reduce the (2,) vector.
        if not target_shape:
            is_contact = jnp.any(contact_flag)
        else:
            # Now contact_flag is (batch, 2). Reduce along the last axis.
            is_contact = jnp.any(contact_flag, axis=-1)

        # Convert the Boolean to float: 1.0 if any contact, 0.0 otherwise.
        contact_value = jnp.where(is_contact, 1.0, 0.0)
        return contact_value

    @property
    def reward_name(self) -> str:
        return "feet_contact_penalty"


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityTrackingReward(ksim.Reward):
    """Reward for tracking the angular velocity."""

    error_scale: float = attrs.field(default=0.25)
    angvel_obs_name: str = attrs.field(default="sensor_observation_base_link_ang_vel")
    command_name: str = attrs.field(default="angular_velocity_step_command")
    norm: xax.NormType = attrs.field(default="l2")

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        if self.angvel_obs_name not in trajectory.obs:
            raise ValueError(f"Observation {self.angvel_obs_name} not found; add it as an observation in your task.")
        ang_vel_error = jnp.square(
            trajectory.command[self.command_name][..., 2] - trajectory.obs[self.angvel_obs_name][..., 2]
        )
        return jnp.exp(-ang_vel_error / self.error_scale)


@attrs.define(frozen=True, kw_only=True)
class TerminationPenalty(ksim.Reward):
    """Penalty for termination."""

    scale: float = attrs.field(default=-1.0)

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        return trajectory.done


class ZbotActor(eqx.Module):
    """Actor for the walking task."""

    mlp: eqx.nn.MLP
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()
    mean_scale: float = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        min_std: float,
        max_std: float,
        var_scale: float,
        mean_scale: float,
    ) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=NUM_INPUTS,
            out_size=NUM_OUTPUTS * 2,
            width_size=64,
            depth=5,
            key=key,
            activation=jax.nn.relu,
        )
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale
        self.mean_scale = mean_scale

    def __call__(
        self,
        joint_pos_n: Array,
        joint_vel_n: Array,
        imu_acc_3: Array,
        imu_gyro_3: Array,
        lin_vel_cmd_2: Array,
        last_action_n: Array,
    ) -> distrax.Normal:
        x_n = jnp.concatenate(
            [joint_pos_n, joint_vel_n, imu_acc_3, imu_gyro_3, lin_vel_cmd_2, last_action_n], axis=-1
        )  # (NUM_INPUTS)

        return self.call_flat_obs(x_n)

    def call_flat_obs(
        self,
        flat_obs_n: Array,
    ) -> distrax.Normal:
        # Split the output into mean and standard deviation.
        prediction_n = self.mlp(flat_obs_n)
        mean_n = prediction_n[..., :NUM_OUTPUTS]
        std_n = prediction_n[..., NUM_OUTPUTS:]

        # Scale the mean.
        mean_n = jnp.tanh(mean_n) * self.mean_scale

        # Softplus and clip to ensure positive standard deviations.
        std_n = jnp.clip((jax.nn.softplus(std_n) + self.min_std) * self.var_scale, max=self.max_std)

        # Return position-only distribution
        return distrax.Normal(mean_n, std_n)


class ZbotCritic(eqx.Module):
    """Critic for the walking task."""

    mlp: eqx.nn.MLP

    def __init__(self, key: PRNGKeyArray) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=NUM_INPUTS,
            out_size=1,  # Always output a single critic value.
            width_size=64,
            depth=5,
            key=key,
            activation=jax.nn.relu,
        )

    def __call__(
        self,
        joint_pos_n: Array,
        joint_vel_n: Array,
        imu_acc_3: Array,
        imu_gyro_3: Array,
        lin_vel_cmd_2: Array,
        last_action_n: Array,
    ) -> Array:
        x_n = jnp.concatenate(
            [joint_pos_n, joint_vel_n, imu_acc_3, imu_gyro_3, lin_vel_cmd_2, last_action_n], axis=-1
        )  # (NUM_INPUTS)
        return self.call_flat_obs(x_n)

    def call_flat_obs(
        self,
        flat_obs_n: Array,
    ) -> Array:
        return self.mlp(flat_obs_n)


class ZbotModel(eqx.Module):
    actor: ZbotActor
    critic: ZbotCritic

    def __init__(self, key: PRNGKeyArray) -> None:
        self.actor = ZbotActor(
            key,
            min_std=0.01,
            max_std=1.0,
            var_scale=1.0,
            mean_scale=1.0,
        )
        self.critic = ZbotCritic(key)


@dataclass
class ZbotWalkingTaskConfig(ZbotTaskConfig):
    """Config for the ZBot walking task."""

    action_scale: float = xax.field(
        value=1.0,
        help="The scale to apply to the actions.",
    )

    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=1e-4,
        help="Learning rate for PPO.",
    )
    max_grad_norm: float = xax.field(
        value=0.5,
        help="Maximum gradient norm for clipping.",
    )
    adam_weight_decay: float = xax.field(
        value=0.0,
        help="Weight decay for the Adam optimizer.",
    )


class ZbotWalkingTask(ZbotTask[ZbotWalkingTaskConfig, ZbotModel]):
    @property
    def get_input_shapes(self) -> list[tuple[int, ...]]:
        return [(NUM_INPUTS,)]

    def get_optimizer(self) -> optax.GradientTransformation:
        """Builds the optimizer.

        This provides a reasonable default optimizer for training PPO models,
        but can be overridden by subclasses who want to do something different.
        """
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            (
                optax.adam(self.config.learning_rate)
                if self.config.adam_weight_decay == 0.0
                else optax.adamw(self.config.learning_rate, weight_decay=self.config.adam_weight_decay)
            ),
        )

        return optimizer

    def get_randomization(self, physics_model: ksim.PhysicsModel) -> list[ksim.Randomization]:
        return [
            ksim.StaticFrictionRandomization(scale_lower=0.5, scale_upper=2.0),
            ksim.JointZeroPositionRandomization(scale_lower=-0.05, scale_upper=0.05),
            ksim.ArmatureRandomization(scale_lower=1.0, scale_upper=1.05),
            ksim.MassMultiplicationRandomization.from_body_name(physics_model, "Top_Brace"),
            ksim.JointDampingRandomization(scale_lower=0.95, scale_upper=1.05),
        ]

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset(scale=0.01),
            ksim.RandomJointVelocityReset(scale=0.01),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            DHJointPositionObservation(),
            DHJointVelocityObservation(),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_acc", noise=0.5),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_gyro", noise=0.2),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="base_link_quat", noise=0.1),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="base_link_vel", noise=0.1),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="base_link_ang_vel", noise=0.1),
            LastActionObservation(noise=0.0),
            ContactObservation.create(
                physics_model=physics_model,
                geom_names=["Left_Foot_collision_box", "Right_Foot_collision_box"],
                contact_group="feet",
                noise=0.0,
            ),
        ]

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> Curriculum:
        """Returns the curriculum for the task."""
        # Using a constant curriculum (max difficulty) as a default
        return ConstantCurriculum(level=1.0)

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [
            ksim.LinearVelocityStepCommand(
                x_range=(0.1, 0.5),
                y_range=(-0.1, 0.1),
                x_fwd_prob=1.0,
                y_fwd_prob=0.5,
                x_zero_prob=0.0,
                y_zero_prob=0.0,
            ),
            ksim.AngularVelocityStepCommand(
                scale=0.0,
                zero_prob=1.0,
            ),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            JointDeviationPenalty(scale=-1.0),
            DHControlPenalty(scale=-0.05),
            DHHealthyReward(scale=0.5),
            DHControlPenalty(scale=-0.01),
            TerminationPenalty(scale=-5.0),
            LinearVelocityTrackingReward(scale=2.0),
            AngularVelocityTrackingReward(scale=0.75),
            FeetContactPenalty(
                contact_obs_key="contact_observation_feet",
                scale=-2.0,
            ),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.RollTooGreatTermination(max_roll=1.04),
            ksim.PitchTooGreatTermination(max_pitch=1.04),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return [
            ksim.PushEvent(
                x_force=0.2,
                y_force=0.2,
                z_force=0.0,
                interval_range=(1.0, 2.0),
            ),
        ]

    def get_model(self, key: PRNGKeyArray) -> ZbotModel:
        return ZbotModel(key)

    def get_initial_carry(self, rng: PRNGKeyArray) -> Array:
        return jnp.zeros(0)

    def _run_actor(
        self,
        model: ZbotModel,
        observations: FrozenDict[str, Array],
        commands: FrozenDict[str, Array],
    ) -> distrax.Normal:
        joint_pos_n = observations["dhjoint_position_observation"]
        joint_vel_n = observations["dhjoint_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        lin_vel_cmd_2 = commands["linear_velocity_step_command"]
        last_action_n = observations["last_action_observation"]

        # Concatenate inputs like the humanoid example
        x_n = jnp.concatenate([joint_pos_n, joint_vel_n, imu_acc_3, imu_gyro_3, lin_vel_cmd_2, last_action_n], axis=-1)

        # Call the actor with a flat observation tensor
        return model.actor.call_flat_obs(x_n)

    def _run_critic(
        self,
        model: ZbotModel,
        observations: FrozenDict[str, Array],
        commands: FrozenDict[str, Array],
    ) -> Array:
        joint_pos_n = observations["dhjoint_position_observation"]
        joint_vel_n = observations["dhjoint_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        lin_vel_cmd_2 = commands["linear_velocity_step_command"]
        last_action_n = observations["last_action_observation"]

        # Concatenate inputs
        x_n = jnp.concatenate([joint_pos_n, joint_vel_n, imu_acc_3, imu_gyro_3, lin_vel_cmd_2, last_action_n], axis=-1)

        # Call critic with flat observation tensor
        return model.critic.call_flat_obs(x_n)

    def get_ppo_variables(
        self,
        model: ZbotModel,
        trajectories: ksim.Trajectory,
        carry: PyTree,
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, PyTree]:
        """Gets the variables required for computing PPO loss."""
        # Extract individual tensors from observations and commands, then reshape for batching
        joint_pos_n = trajectories.obs["dhjoint_position_observation"]  # (..., N)
        joint_vel_n = trajectories.obs["dhjoint_velocity_observation"]  # (..., N)
        imu_acc_3 = trajectories.obs["sensor_observation_imu_acc"]  # (..., 3)
        imu_gyro_3 = trajectories.obs["sensor_observation_imu_gyro"]  # (..., 3)
        lin_vel_cmd_2 = trajectories.command["linear_velocity_step_command"]  # (..., 2)
        last_action_n = trajectories.obs["last_action_observation"]  # (..., N)

        # Concatenate inputs to create a single tensor for each batch item
        flat_obs = jnp.concatenate(
            [joint_pos_n, joint_vel_n, imu_acc_3, imu_gyro_3, lin_vel_cmd_2, last_action_n], axis=-1
        )

        # Call actor and critic directly with batched inputs
        action_dist = model.actor.call_flat_obs(flat_obs)
        log_probs = action_dist.log_prob(trajectories.action)

        values = model.critic.call_flat_obs(flat_obs).squeeze(-1)

        # Return PPO variables and carry
        return ksim.PPOVariables(log_probs=log_probs, values=values), carry

    def sample_action(
        self,
        model: ZbotModel,
        carry: Array,
        physics_model: ksim.PhysicsModel,
        physics_state: PhysicsState,
        observations: FrozenDict[str, Array],
        commands: FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> ksim.Action:
        action_dist_n = self._run_actor(model, observations, commands)
        action_n = action_dist_n.sample(seed=rng) * self.config.action_scale

        return ksim.Action(action=action_n, carry=jnp.zeros(0), aux_outputs=None)


if __name__ == "__main__":
    # python -m ksim_zbot.zbot2.walking run_environment=True
    ZbotWalkingTask.launch(
        ZbotWalkingTaskConfig(
            num_envs=4096,
            batch_size=256,
            num_passes=10,
            epochs_per_log_step=1,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            max_action_latency=0.005,
            min_action_latency=0.0,
            log_full_trajectory_every_n_steps=5,
            log_full_trajectory_on_first_step=True,
            save_every_n_steps=5,
            rollout_length_seconds=5.0,
            # PPO parameters
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.001,
            learning_rate=3e-4,
            clip_param=0.3,
            max_grad_norm=1.0,
            export_for_inference=True,
        ),
    )
