# mypy: disable-error-code="override"
"""Defines simple task for training a walking policy for Z-Bot."""

from dataclasses import dataclass

import attrs
import jax
import jax.numpy as jnp
import ksim
import xax
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import JointMetadataOutput

from .standing import ZbotStandingTask, ZbotStandingTaskConfig, LastActionObservation, HistoryObservation


@attrs.define(frozen=True, kw_only=True)
class UpwardReward(ksim.Reward):
    """Incentives forward movement."""

    velocity_clip: float = attrs.field(default=10.0)

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        # Just try to maximize the velocity in the Z direction.
        z_delta = jnp.clip(trajectory.qvel[..., 2], 0, self.velocity_clip)
        return z_delta


@attrs.define(frozen=True, kw_only=True)
class StationaryPenalty(ksim.Reward):
    """Incentives staying in place laterally."""

    norm: xax.NormType = attrs.field(default="l2")

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        return xax.get_norm(trajectory.qvel[..., :2], self.norm).sum(axis=-1)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AuxOutputs:
    log_probs: Array
    values: Array


@dataclass
class ZbotJumpingTaskConfig(ZbotStandingTaskConfig):
    pass


class ZbotJumpingTask(ZbotStandingTask[ZbotJumpingTaskConfig]):
    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            UpwardReward(scale=0.5),
            StationaryPenalty(scale=-0.1),
            ksim.ActuatorForcePenalty(scale=-0.01),
            ksim.LinearVelocityZPenalty(scale=-0.01),
            ksim.AngularVelocityXYPenalty(scale=-0.01),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return [
            ksim.PushEvent(
                x_force=0.1,
                y_force=0.1,
                z_force=0.0,
                interval_range=(1, 5),
            ),
        ]

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: dict[str, JointMetadataOutput] | None = None,
    ) -> ksim.Actuators:
        if self.config.use_mit_actuators:
            if metadata is None:
                raise ValueError("Metadata is required for MIT actuators")
            return ksim.MITPositionVelocityActuators(
                physics_model,
                metadata,
                pos_action_noise=0.1,
                vel_action_noise=0.1,
                pos_action_noise_type="gaussian",
                vel_action_noise_type="gaussian",
                ctrl_clip=None,
            )
        else:
            return ksim.TorqueActuators()

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            ksim.JointPositionObservation(noise=0.0),
            ksim.JointVelocityObservation(noise=0.0),
            ksim.SensorObservation.create(physics_model, "IMU_acc", noise=0.0),
            ksim.SensorObservation.create(physics_model, "IMU_gyro", noise=0.0),
            ksim.ActuatorForceObservation(),
            LastActionObservation(noise=0.0),
            HistoryObservation(),
        ]

    def get_initial_carry(self, rng: PRNGKeyArray) -> Array:
        from .standing import HISTORY_LENGTH, SINGLE_STEP_HISTORY_SIZE
        return jnp.zeros(HISTORY_LENGTH * SINGLE_STEP_HISTORY_SIZE)


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m ksim_zbot.zbot2.jumping
    # To visualize the environment, use the following command:
    #   python -m ksim_zbot.zbot2.jumping run_environment=True
    ZbotJumpingTask.launch(
        ZbotJumpingTaskConfig(
            num_envs=2048,
            batch_size=64,
            num_passes=8,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.002,
            min_action_latency=0.0,
            valid_every_n_steps=25,
            valid_every_n_seconds=300,
            valid_first_n_steps=0,
            rollout_length_seconds=10.0,
            # PPO parameters
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.001,
            learning_rate=3e-4,
            clip_param=0.3,
            max_grad_norm=1.0,
            use_mit_actuators=True,
        ),
    )
