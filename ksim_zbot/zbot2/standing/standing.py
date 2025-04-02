"""Defines simple task for training a walking policy for Z-Bot."""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, List, Optional, TypedDict, TypeVar

import attrs
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import ksim
import mujoco
import optax
import xax
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import JointMetadataOutput
from ksim.actuators import Actuators, NoiseType
from ksim.types import PhysicsData
from mujoco import mjx
from mujoco_scenes.mjcf import load_mjmodel
from scipy.interpolate import CubicSpline
from xax.nn.export import export
from xax.utils.types.frozen_dict import FrozenDict

OBS_SIZE = 20 * 2 + 3 + 3 + 20  # position + velocity + imu_acc + imu_gyro + last_action
CMD_SIZE = 2
NUM_OUTPUTS = 20  # position only for FeetechActuators (not position + velocity)

SINGLE_STEP_HISTORY_SIZE = NUM_OUTPUTS + OBS_SIZE + CMD_SIZE

HISTORY_LENGTH = 0

NUM_INPUTS = (OBS_SIZE + CMD_SIZE) + SINGLE_STEP_HISTORY_SIZE * HISTORY_LENGTH

Config = TypeVar("Config", bound="ZbotStandingTaskConfig")


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AuxOutputs:
    log_probs: Array
    values: Array


@attrs.define(frozen=True)
class HistoryObservation(ksim.Observation):
    def observe(self, state: ksim.RolloutVariables, rng: PRNGKeyArray) -> Array:
        if not isinstance(state.carry, Array):
            raise ValueError("Carry is not a history array")
        return state.carry


ErrorGainData = List[dict[str, float]]


class FeetechParams(TypedDict):
    sysid: str
    max_torque: float
    armature: float
    frictionloss: float
    damping: float
    vin: float
    kt: float
    R: float
    error_gain_data: ErrorGainData


class FeetechActuators(Actuators):
    """Feetech actuator controller using padded JAX arrays (dynamic slice fix)."""

    def __init__(
        self,
        max_torque: Array,
        kp: Array,
        kd: Array,
        error_gain_data: List[ErrorGainData], # Mandatory
        action_noise: float = 0.0,
        action_noise_type: NoiseType = "none",
        torque_noise: float = 0.0,
        torque_noise_type: NoiseType = "none",
    ) -> None:
        self.max_torque = max_torque
        self.kp = kp
        self.kd = kd
        num_outputs = kp.shape[0]

        if len(error_gain_data) != num_outputs:
            raise ValueError(f"Length of error_gain_data ({len(error_gain_data)}) must match number of actuators ({num_outputs}).")

        temp_knots_list: list[np.ndarray] = []
        temp_coeffs_list: list[np.ndarray] = []
        actual_knot_counts: list[int] = []

        # --- Process data (same as before) ---
        for i, ed in enumerate(error_gain_data):
            # ... (validation checks: None, len < 2, duplicates) ...
            if ed is None or len(ed) < 2: raise ValueError(f"Actuator {i}: Invalid error_gain_data.")
            ed_sorted = sorted(ed, key=lambda d: d["pos_err"])
            x_vals = [d["pos_err"] for d in ed_sorted]
            if len(set(x_vals)) != len(x_vals): raise ValueError(f"Actuator {i}: Duplicate pos_err.")
            y_vals = [d["error_gain"] for d in ed_sorted]

            cs = CubicSpline(x_vals, y_vals, extrapolate=True)
            knots = np.array(cs.x, dtype=np.float32)
            coeffs = np.array(cs.c, dtype=np.float32)
            if knots.size < 2: raise ValueError(f"Actuator {i}: Spline fitting < 2 knots.")

            temp_knots_list.append(knots)
            temp_coeffs_list.append(coeffs)
            actual_knot_counts.append(knots.size)

        max_num_knots = max(actual_knot_counts) if actual_knot_counts else 0
        if max_num_knots < 2: raise ValueError("Valid spline data requires at least 2 knots.")
        max_num_coeffs_intervals = max_num_knots - 1

        # --- Create padded JAX arrays ---
        # ***** CHANGE 1: Use jnp.inf for knot padding *****
        knot_padding_value = jnp.inf
        coeff_padding_value = jnp.nan # Coeff padding can be NaN or 0

        self.stacked_knots = jnp.full((num_outputs, max_num_knots), knot_padding_value, dtype=jnp.float32)
        self.stacked_coeffs = jnp.full((num_outputs, 4, max_num_coeffs_intervals), coeff_padding_value, dtype=jnp.float32)
        self.knot_counts = jnp.array(actual_knot_counts, dtype=jnp.int32)

        # --- Fill padded arrays (same as before) ---
        for i in range(num_outputs):
            k = temp_knots_list[i]
            c = temp_coeffs_list[i]
            count = actual_knot_counts[i] # Use Python int here for slicing numpy array k
            self.stacked_knots = self.stacked_knots.at[i, :count].set(jnp.array(k))
            self.stacked_coeffs = self.stacked_coeffs.at[i, :, :(count - 1)].set(jnp.array(c))

        # --- Store other parameters (same as before) ---
        self.has_spline_data_flags = jnp.ones(num_outputs, dtype=bool)
        self.default_error_gain = jnp.ones(num_outputs, dtype=jnp.float32)
        self.action_noise = action_noise
        self.action_noise_type = action_noise_type
        self.torque_noise = torque_noise
        self.torque_noise_type = torque_noise_type


    # --- Spline evaluation: Remove dynamic slice before searchsorted ---
    def _eval_spline(self, x_val_sat, knots, coeffs, knot_count):
        """Evaluate spline using SciPy format on potentially padded arrays (dynamic slice fix)."""

        # ***** CHANGE 2: Search on the full padded knot array *****
        # `searchsorted` works correctly with finite x_val_sat and jnp.inf padding
        search_result = jnp.searchsorted(knots, x_val_sat)

        # Clip the index result to the valid range for *coefficients* [0, knot_count - 2]
        # This handles cases where search_result might be >= knot_count
        idx = jnp.clip(search_result - 1, 0, knot_count - 2)

        # Calculate difference from the knot (use idx which is safe)
        dx = x_val_sat - knots[idx]

        # Evaluate polynomial (same as before)
        spline_value = coeffs[0, idx] * dx**3 + coeffs[1, idx] * dx**2 + coeffs[2, idx] * dx + coeffs[3, idx]
        return spline_value

    # --- get_ctrl remains the same as the previous version ---
    def get_ctrl(self, action: Array, physics_data: PhysicsData, rng: PRNGKeyArray) -> Array:
        """Compute torque control using Feetech parameters and mandatory cubic spline (JAX friendly)."""
        pos_rng, tor_rng = jax.random.split(rng)
        current_pos = physics_data.qpos[7:]
        current_vel = physics_data.qvel[6:]
        pos_error = action - current_pos
        vel_error = -current_vel

        def process_single_joint(err, k, c, kc, flag, default_eg):
            abs_err = jnp.abs(err)
            x_clamped = jnp.clip(abs_err, k[0], k[kc - 1]) # Use knot_count 'kc' for safe indexing

            def eval_spline_branch():
                 return self._eval_spline(x_clamped, k, c, kc)
            def default_gain_branch():
                 return default_eg

            result = jax.lax.cond(flag, eval_spline_branch, default_gain_branch)
            return result

        error_gain = jax.vmap(process_single_joint)(
            pos_error,
            self.stacked_knots,
            self.stacked_coeffs,
            self.knot_counts,
            self.has_spline_data_flags,
            self.default_error_gain
        )

        duty = self.kp * error_gain * pos_error + self.kd * vel_error
        torque = jnp.clip(
            self.add_noise(self.torque_noise, self.torque_noise_type, duty * self.max_torque, tor_rng),
            -self.max_torque,
            self.max_torque,
        )
        return torque

    def get_default_action(self, physics_data: PhysicsData) -> Array:
        return physics_data.qpos[7:]


@attrs.define(frozen=True, kw_only=True)
class JointDeviationPenalty(ksim.Reward):
    """Penalty for joint deviations."""

    norm: xax.NormType = attrs.field(default="l2")
    joint_targets: tuple[float, ...] = attrs.field(
        default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        diff = trajectory.qpos[..., 7:] - jnp.array(self.joint_targets)
        return xax.get_norm(diff, self.norm).sum(axis=-1)


@attrs.define(frozen=True)
class JointPositionObservation(ksim.Observation):
    noise: float = attrs.field(default=0.0)
    default_targets: tuple[float, ...] = attrs.field(
        default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )

    def observe(self, rollout_state: ksim.RolloutVariables, rng: PRNGKeyArray) -> Array:
        qpos = rollout_state.physics_state.data.qpos[7:]  # (N,)
        diff = qpos - jnp.array(self.default_targets)
        return diff

    def add_noise(self, observation: Array, rng: PRNGKeyArray) -> Array:
        return observation + jax.random.normal(rng, observation.shape) * self.noise


@attrs.define(frozen=True)
class LastActionObservation(ksim.Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, rollout_state: ksim.RolloutVariables, rng: PRNGKeyArray) -> Array:
        return rollout_state.physics_state.most_recent_action

    def add_noise(self, observation: Array, rng: PRNGKeyArray) -> Array:
        return observation + jax.random.normal(rng, observation.shape) * self.noise


@attrs.define(frozen=True, kw_only=True)
class ResetDefaultJointPosition(ksim.Reset):
    """Resets the joint positions of the robot to random values."""

    default_targets: tuple[float, ...] = attrs.field(
        default=(
            # xyz
            0.0,
            0.0,
            0.41,  # This is the starting height (Z coordinate)
            # quat
            1.0,
            0.0,
            0.0,
            0.0,
            # qpos - 20 elements for zbot-6dof-feet's joint positions
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    )

    def __call__(self, data: ksim.PhysicsData, rng: PRNGKeyArray) -> ksim.PhysicsData:
        qpos = data.qpos
        match type(data):
            case mujoco.MjData:
                qpos[:] = self.default_targets
            case mjx.Data:
                qpos = qpos.at[:].set(self.default_targets)
        return ksim.utils.mujoco.update_data_field(data, "qpos", qpos)


@attrs.define(frozen=True, kw_only=True)
class FeetSlipPenalty(ksim.Reward):
    """Penalty for feet slipping."""

    norm: xax.NormType = attrs.field(default="l2")
    observation_name: str = attrs.field(default="feet_contact_observation")
    command_name: str = attrs.field(default="linear_velocity_step_command")
    com_vel_obs_name: str = attrs.field(default="center_of_mass_velocity_observation")
    command_vel_scale: float = attrs.field(default=0.02)

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        if self.observation_name not in trajectory.obs:
            raise ValueError(f"Observation {self.observation_name} not found; add it as an observation in your task.")
        contact = trajectory.obs[self.observation_name]
        com_vel = trajectory.obs[self.com_vel_obs_name][..., :2]
        return (xax.get_norm(com_vel, self.norm) * contact).sum(axis=-1)


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
            out_size=NUM_OUTPUTS * 2,  # Still need mean and std for each output
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
        history_n: Array,
    ) -> distrax.Normal:
        x_n = jnp.concatenate(
            [
                joint_pos_n,
                joint_vel_n,
                imu_acc_3,
                imu_gyro_3,
                lin_vel_cmd_2,
                last_action_n,
                # history_n,
            ],
            axis=-1,
        )  # (NUM_INPUTS)

        return self.call_flat_obs(x_n)

    def call_flat_obs(
        self,
        flat_obs_n: Array,
    ) -> distrax.Normal:
        prediction_n = self.mlp(flat_obs_n)
        mean_n = prediction_n[..., :NUM_OUTPUTS]
        std_n = prediction_n[..., NUM_OUTPUTS:]

        # Scale the mean.
        mean_n = jnp.tanh(mean_n) * self.mean_scale

        # Softplus and clip to ensure positive standard deviations.
        std_n = jnp.clip((jax.nn.softplus(std_n) + self.min_std) * self.var_scale, max=self.max_std)

        return distrax.Normal(mean_n, std_n)


class ZbotCritic(eqx.Module):
    """Critic for the standing task."""

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
        history_n: Array,
    ) -> Array:
        x_n = jnp.concatenate(
            [
                joint_pos_n,
                joint_vel_n,
                imu_acc_3,
                imu_gyro_3,
                lin_vel_cmd_2,
                last_action_n,
                # history_n,
            ],
            axis=-1,
        )  # (NUM_INPUTS)
        return self.mlp(x_n)


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
class ZbotStandingTaskConfig(ksim.PPOConfig):
    """Config for the Z-Bot walking task."""

    robot_urdf_path: str = xax.field(
        value="ksim_zbot/kscale-assets/zbot-6dof-feet/",
        help="The path to the assets directory for the robot.",
    )

    actuator_params_path: str = xax.field(
        value="ksim_zbot/kscale-assets/actuators/feetech/",
        help="The path to the assets directory for feetech actuator models",
    )

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

    use_mit_actuators: bool = xax.field(
        value=False,
        help="Whether to use the MIT actuator model, where the actions are position commands",
    )

    # Rendering parameters.
    render_track_body_id: int | None = xax.field(
        value=None,
        help="The body id to track with the render camera.",
    )

    # Checkpointing parameters.
    export_for_inference: bool = xax.field(
        value=True,
        help="Whether to export the model for inference.",
    )


class ZbotStandingTask(ksim.PPOTask[ZbotStandingTaskConfig], Generic[Config]):
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

    def get_mujoco_model(self) -> mujoco.MjModel:
        # mjcf_path = (Path(self.config.robot_urdf_path) / "scene.mjcf").resolve().as_posix()
        mjcf_path = (Path(self.config.robot_urdf_path) / "robot.mjcf").resolve().as_posix()
        print(f"Loading MJCF model from {mjcf_path}")
        mj_model = load_mjmodel(mjcf_path, scene="smooth")

        mj_model.opt.timestep = jnp.array(self.config.dt)
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        mj_model.opt.disableflags = mjx.DisableBit.EULERDAMP
        mj_model.opt.solver = mjx.SolverType.CG

        sts3215_params, sts3250_params = self._load_feetech_params()

        required_keys = ["damping", "armature", "frictionloss", "max_torque"]
        for key in required_keys:
            if key not in sts3215_params:
                raise ValueError(f"Missing required key '{key}' in sts3215 parameters.")
            if key not in sts3250_params:
                raise ValueError(f"Missing required key '{key}' in sts3250 parameters.")

        # Apply servo-specific parameters based on joint name suffix
        for i in range(mj_model.njnt):
            joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name is None or not any(suffix in joint_name for suffix in ["_15", "_50"]):
                continue

            dof_id = mj_model.jnt_dofadr[i]

            # Apply parameters based on the joint suffix
            if "_15" in joint_name:  # STS3215 servos (arms)
                mj_model.dof_damping[dof_id] = sts3215_params["damping"]
                mj_model.dof_armature[dof_id] = sts3215_params["armature"]
                mj_model.dof_frictionloss[dof_id] = sts3215_params["frictionloss"]

                # Get base name for actuator (remove the _15 suffix)
                base_name = joint_name.rsplit("_", 1)[0]
                actuator_name = f"{base_name}_15_ctrl"

                actuator_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                if actuator_id >= 0:
                    max_torque = float(sts3215_params["max_torque"])
                    mj_model.actuator_forcerange[actuator_id, :] = [
                        -max_torque,
                        max_torque,
                    ]

            elif "_50" in joint_name:  # STS3250 servos (legs)
                mj_model.dof_damping[dof_id] = sts3250_params["damping"]
                mj_model.dof_armature[dof_id] = sts3250_params["armature"]
                mj_model.dof_frictionloss[dof_id] = sts3250_params["frictionloss"]

                # Get base name for actuator (remove the _50 suffix)
                base_name = joint_name.rsplit("_", 1)[0]
                actuator_name = f"{base_name}_50_ctrl"

                actuator_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                if actuator_id >= 0:
                    max_torque = float(sts3250_params["max_torque"])
                    mj_model.actuator_forcerange[actuator_id, :] = [
                        -max_torque,
                        max_torque,
                    ]

        return mj_model

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> dict[str, JointMetadataOutput]:
        metadata = asyncio.run(ksim.get_mujoco_model_metadata(self.config.robot_urdf_path, cache=False))
        if metadata.joint_name_to_metadata is None:
            raise ValueError("Joint metadata is not available")
        return metadata.joint_name_to_metadata

    def _load_feetech_params(self) -> tuple[FeetechParams, FeetechParams]:
        params_path = Path(self.config.actuator_params_path)
        params_file_3215 = params_path / "sts3215_12v_params.json"
        params_file_3250 = params_path / "sts3250_params.json"

        if not params_file_3215.exists():
            raise ValueError(
                f"Feetech parameters file '{params_file_3215}' not found. Please ensure it exists in '{params_path}'."
            )
        if not params_file_3250.exists():
            raise ValueError(
                f"Feetech parameters file '{params_file_3250}' not found. Please ensure it exists in '{params_path}'."
            )

        with open(params_file_3215, "r") as f:
            params_3215: FeetechParams = json.load(f)
        with open(params_file_3250, "r") as f:
            params_3250: FeetechParams = json.load(f)
        return params_3215, params_3250

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: dict[str, JointMetadataOutput] | None = None,
    ) -> ksim.Actuators:
        if metadata is not None:
            joint_names = sorted(metadata.keys())

            num_joints = len(joint_names)
            max_torque_arr = jnp.zeros(num_joints)
            kp_arr = jnp.zeros(num_joints)
            kd_arr = jnp.zeros(num_joints)

            sts3215_params, sts3250_params = self._load_feetech_params()
            required_keys = ["max_torque", "error_gain_data"]
            for key in required_keys:
                if key not in sts3215_params:
                    raise ValueError(f"Missing required key '{key}' in sts3215 parameters.")
                if key not in sts3250_params:
                    raise ValueError(f"Missing required key '{key}' in sts3250 parameters.")

            if not isinstance(sts3215_params["error_gain_data"], list):
                raise ValueError("sts3215_params['error_gain_data'] must be a list.")
            if not isinstance(sts3250_params["error_gain_data"], list):
                raise ValueError("sts3250_params['error_gain_data'] must be a list.")

            # Build a list of error_gain_data (one entry per joint)
            error_gain_data_list: List[ErrorGainData] = []

            for i, joint_name in enumerate(joint_names):
                joint_meta = metadata[joint_name]
                if "_15" in joint_name:
                    max_torque_arr = max_torque_arr.at[i].set(sts3215_params["max_torque"])
                    error_gain_data_list.append(sts3215_params["error_gain_data"])
                elif "_50" in joint_name:
                    max_torque_arr = max_torque_arr.at[i].set(sts3250_params["max_torque"])
                    error_gain_data_list.append(sts3250_params["error_gain_data"])
                else:
                    raise ValueError(f"Invalid joint name: {joint_name}")

                if joint_meta.kp is None:
                    raise ValueError(f"kp is not available for joint {joint_name}")
                if joint_meta.kd is None:
                    raise ValueError(f"kd is not available for joint {joint_name}")

                try:
                    kp_val = float(joint_meta.kp)
                    kd_val = float(joint_meta.kd)
                except ValueError as e:
                    raise ValueError(f"Could not convert kp/kd gains to a float for joint {joint_name}: {e}")

                kp_arr = kp_arr.at[i].set(kp_val)
                kd_arr = kd_arr.at[i].set(kd_val)
        else:
            raise ValueError("Metadata is not available")
        return FeetechActuators(
            max_torque=max_torque_arr,
            kp=kp_arr,
            kd=kd_arr,
            error_gain_data=error_gain_data_list,
            action_noise=0.0,
            action_noise_type="none",
            torque_noise=0.0,
            torque_noise_type="none",
        )

    def get_randomization(self, physics_model: ksim.PhysicsModel) -> list[ksim.Randomization]:
        return [
            ksim.StaticFrictionRandomization(scale_lower=0.5, scale_upper=2.0),
            ksim.JointZeroPositionRandomization(scale_lower=-0.05, scale_upper=0.05),
            ksim.ArmatureRandomization(scale_lower=1.0, scale_upper=1.05),
            ksim.MassMultiplicationRandomization.from_body_name(physics_model, "Top_Brace"),
            ksim.JointDampingRandomization(scale_lower=0.95, scale_upper=1.05),
            # ksim.FloorFrictionRandomization.from_body_name(
            #     model=physics_model,
            #     scale_lower=0.2,
            #     scale_upper=0.6,
            #     floor_body_name="floor",
            # ),
        ]

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomBaseVelocityXYReset(scale=0.01),
            ksim.RandomJointPositionReset(scale=0.02),
            ksim.RandomJointVelocityReset(scale=0.02),
            ResetDefaultJointPosition(
                default_targets=(
                    0.0,
                    0.0,
                    0.40,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            ),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return [
            # ksim.PushEvent(
            #     x_force=0.2,
            #     y_force=0.2,
            #     z_force=0.0,
            #     interval_range=(1.0, 2.0),
            # ),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            # ksim.JointPositionObservation(noise=0.02),
            JointPositionObservation(
                noise=0.05,
                default_targets=(
                    # right arm
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    # left arm
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    # right leg
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ),
            ),
            ksim.JointVelocityObservation(noise=0.5),
            ksim.ActuatorForceObservation(),
            ksim.SensorObservation.create(physics_model, "IMU_2_acc", noise=0.5),
            ksim.SensorObservation.create(physics_model, "IMU_2_gyro", noise=0.2),
            LastActionObservation(noise=0.0),
            HistoryObservation(),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [
            ksim.LinearVelocityStepCommand(
                x_range=(0.0, 0.0),
                y_range=(0.0, 0.0),
                x_fwd_prob=0.8,
                y_fwd_prob=0.5,
                x_zero_prob=0.2,
                y_zero_prob=0.8,
            ),
            ksim.AngularVelocityStepCommand(
                scale=0.0,
                zero_prob=1.0,
            ),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            JointDeviationPenalty(
                scale=-0.3,
                joint_targets=(
                    # right arm
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    # left arm
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    # right leg
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ),
            ),
            DHControlPenalty(scale=-0.05),
            DHHealthyReward(scale=0.5),
            ksim.ActuatorForcePenalty(scale=-0.01),
            ksim.BaseHeightReward(scale=1.0, height_target=0.4),
            ksim.LinearVelocityTrackingPenalty(command_name="linear_velocity_step_command", scale=-0.05),
            ksim.AngularVelocityTrackingPenalty(command_name="angular_velocity_step_command", scale=-0.05),
            # FeetSlipPenalty(scale=-0.01),
            # ksim.ActionSmoothnessPenalty(scale=-0.01),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.RollTooGreatTermination(max_roll=2.04),
            ksim.PitchTooGreatTermination(max_pitch=2.04),
        ]

    def get_model(self, key: PRNGKeyArray) -> ZbotModel:
        return ZbotModel(key)

    def get_initial_carry(self, rng: PRNGKeyArray) -> Array:
        return jnp.zeros(HISTORY_LENGTH * SINGLE_STEP_HISTORY_SIZE)

    def _run_actor(
        self,
        model: ZbotModel,
        observations: FrozenDict[str, Array],
        commands: FrozenDict[str, Array],
    ) -> distrax.Normal:
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        imu_acc_3 = observations["IMU_2_acc_obs"]
        imu_gyro_3 = observations["IMU_2_gyro_obs"]
        lin_vel_cmd_2 = commands["linear_velocity_step_command"]
        last_action_n = observations["last_action_observation"]
        history_n = observations["history_observation"]
        return model.actor(joint_pos_n, joint_vel_n, imu_acc_3, imu_gyro_3, lin_vel_cmd_2, last_action_n, history_n)

    def _run_critic(
        self,
        model: ZbotModel,
        observations: FrozenDict[str, Array],
        commands: FrozenDict[str, Array],
    ) -> Array:
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        imu_acc_3 = observations["IMU_2_acc_obs"]
        imu_gyro_3 = observations["IMU_2_gyro_obs"]
        lin_vel_cmd_2 = commands["linear_velocity_step_command"]
        last_action_n = observations["last_action_observation"]
        history_n = observations["history_observation"]
        return model.critic(joint_pos_n, joint_vel_n, imu_acc_3, imu_gyro_3, lin_vel_cmd_2, last_action_n, history_n)

    def get_on_policy_log_probs(
        self,
        model: ZbotModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> Array:
        if trajectories.aux_outputs is None:
            raise ValueError("No aux outputs found in trajectories")
        return trajectories.aux_outputs.log_probs

    def get_on_policy_values(
        self,
        model: ZbotModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> Array:
        if trajectories.aux_outputs is None:
            raise ValueError("No aux outputs found in trajectories")
        return trajectories.aux_outputs.values

    def get_log_probs(
        self,
        model: ZbotModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> tuple[Array, Array]:
        # Vectorize over both batch and time dimensions.
        par_fn = jax.vmap(self._run_actor, in_axes=(None, 0, 0))
        action_dist_btn = par_fn(model, trajectories.obs, trajectories.command)

        # Compute the log probabilities of the trajectory's actions according
        # to the current policy, along with the entropy of the distribution.
        action_btn = trajectories.action
        log_probs_btn = action_dist_btn.log_prob(action_btn)
        entropy_btn = action_dist_btn.entropy()

        return log_probs_btn, entropy_btn

    def get_values(
        self,
        model: ZbotModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> Array:
        # Vectorize over both batch and time dimensions.
        par_fn = jax.vmap(self._run_critic, in_axes=(None, 0, 0))
        values_bt1 = par_fn(model, trajectories.obs, trajectories.command)

        # Remove the last dimension.
        return values_bt1.squeeze(-1)

    def sample_action(
        self,
        model: ZbotModel,
        carry: Array,
        physics_model: ksim.PhysicsModel,
        observations: FrozenDict[str, Array],
        commands: FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> tuple[Array, Array, AuxOutputs]:
        action_dist_n = self._run_actor(model, observations, commands)
        action_n = action_dist_n.sample(seed=rng)
        action_log_prob_n = action_dist_n.log_prob(action_n)

        critic_n = self._run_critic(model, observations, commands)
        value_n = critic_n.squeeze(-1)

        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        imu_acc_3 = observations["IMU_2_acc_obs"]
        imu_gyro_3 = observations["IMU_2_gyro_obs"]
        lin_vel_cmd_2 = commands["linear_velocity_step_command"]
        last_action_n = observations["last_action_observation"]
        history_n = jnp.concatenate(
            [
                joint_pos_n,
                joint_vel_n,
                imu_acc_3,
                imu_gyro_3,
                lin_vel_cmd_2,
                last_action_n,
                action_n,
            ],
            axis=-1,
        )

        if HISTORY_LENGTH > 0:
            # Roll the history by shifting the existing history and adding the new data
            carry_reshaped = carry.reshape(HISTORY_LENGTH, SINGLE_STEP_HISTORY_SIZE)
            shifted_history = jnp.roll(carry_reshaped, shift=-1, axis=0)
            new_history = shifted_history.at[HISTORY_LENGTH - 1].set(history_n)
            history_n = new_history.reshape(-1)
        else:
            history_n = jnp.zeros(0)

        return action_n, history_n, AuxOutputs(log_probs=action_log_prob_n, values=value_n)

    def make_export_model(self, model: ZbotModel, stochastic: bool = False, batched: bool = False) -> Callable:
        """Makes a callable inference function that directly takes a flattened input vector and returns an action.

        Returns:
            A tuple containing the inference function and the size of the input vector.
        """

        def deterministic_model_fn(obs: Array) -> Array:
            return model.actor.call_flat_obs(obs).mode()

        def stochastic_model_fn(obs: Array) -> Array:
            distribution = model.actor.call_flat_obs(obs)
            return distribution.sample(seed=jax.random.PRNGKey(0))

        if stochastic:
            model_fn = stochastic_model_fn
        else:
            model_fn = deterministic_model_fn

        if batched:

            def batched_model_fn(obs: Array) -> Array:
                return jax.vmap(model_fn)(obs)

            return batched_model_fn

        return model_fn

    def on_after_checkpoint_save(self, ckpt_path: Path, state: xax.State) -> xax.State:
        state = super().on_after_checkpoint_save(ckpt_path, state)

        model: ZbotModel = self.load_checkpoint(ckpt_path, part="model")

        model_fn = self.make_export_model(model, stochastic=False, batched=True)

        input_shapes: list[tuple[int, ...]] = [(NUM_INPUTS,)]

        export(
            model_fn,
            input_shapes,
            ckpt_path.parent / "tf_model",
        )

        return state





if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m ksim_zbot.zbot2.standing
    # To visualize the environment, use the following command:
    #   python -m ksim_zbot.zbot2.standing \
    #       run_environment=True eval_mode=True valid_every_n_steps=1000
    ZbotStandingTask.launch(
        ZbotStandingTaskConfig(
            num_envs=4096,
            batch_size=256,
            num_passes=10,
            epochs_per_log_step=1,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            max_action_latency=0.005,
            min_action_latency=0.0,
            valid_every_n_steps=5,
            valid_first_n_steps=0,
            save_every_n_steps=5,
            rollout_length_seconds=5.0,
            # PPO parameters
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.001,
            learning_rate=1e-4,
            clip_param=0.3,
            max_grad_norm=1.0,
            export_for_inference=True,
        ),
    )
