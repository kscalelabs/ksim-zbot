"""Common definitions and utilities for Z-Bot tasks."""

import abc
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, TypedDict, TypeVar

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import mujoco
import numpy as np
import optax
import xax
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import JointMetadataOutput
from ksim.actuators import Actuators, NoiseType
from ksim.types import PhysicsData, PhysicsState
from mujoco import mjx
from mujoco_scenes.mjcf import load_mjmodel
from scipy.interpolate import CubicSpline
from xax.nn.export import export
from xax.utils.types.frozen_dict import FrozenDict


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AuxOutputs:
    log_probs: Array
    values: Array


@dataclass
class ZbotTaskConfig(ksim.PPOConfig):
    """Config for the Z-Bot walking task."""

    robot_urdf_path: str = xax.field(
        value="ksim_zbot/kscale-assets/zbot-6dof-feet/",
        help="The path to the assets directory for the robot.",
    )

    actuator_params_path: str = xax.field(
        value="ksim_zbot/kscale-assets/actuators/feetech/",
        help="The path to the assets directory for feetech actuator models",
    )

    # Checkpointing parameters.
    export_for_inference: bool = xax.field(
        value=True,
        help="Whether to export the model for inference.",
    )

    # Rendering parameters.
    render_track_body_id: int | None = xax.field(
        value=None,
        help="The body id to track with the render camera.",
    )

    render_distance: float = xax.field(
        value=1.5,
        help="The distance to the render camera from the robot.",
    )


Config = TypeVar("Config", bound=ZbotTaskConfig)


ErrorGainData = list[dict[str, float]]


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
        max_torque_j: Array,
        kp_j: Array,
        kd_j: Array,
        error_gain_data_j: list[ErrorGainData],  # Mandatory
        action_noise: float = 0.0,
        action_noise_type: NoiseType = "none",
        torque_noise: float = 0.0,
        torque_noise_type: NoiseType = "none",
    ) -> None:
        self.max_torque_j = max_torque_j
        self.kp_j = kp_j
        self.kd_j = kd_j
        j = kp_j.shape[0]  # num outputs

        if len(error_gain_data_j) != j:
            raise ValueError(
                f"Length of error_gain_data ({len(error_gain_data_j)}) must match number of actuators ({j})."
            )

        temp_knots_j: list[np.ndarray] = []
        temp_coeffs_j: list[np.ndarray] = []
        knot_counts_j: list[int] = []

        for i, ed in enumerate(error_gain_data_j):
            # Checks for valid error_gain_data
            if ed is None or len(ed) < 3:
                raise ValueError(f"Actuator {i}: Not enough error_gain_data.")
            ed_sorted = sorted(ed, key=lambda d: d["pos_err"])
            x_vals = [d["pos_err"] for d in ed_sorted]
            if len(set(x_vals)) != len(x_vals):
                raise ValueError(f"Actuator {i}: Duplicate pos_err.")
            y_vals = [d["error_gain"] for d in ed_sorted]

            # Fit cubic spline
            cs = CubicSpline(x_vals, y_vals, extrapolate=True)
            knots = np.array(cs.x, dtype=np.float32)
            coeffs = np.array(cs.c, dtype=np.float32)
            if knots.size < 2:
                raise ValueError(f"Actuator {i}: Spline fitting < 2 knots.")

            temp_knots_j.append(knots)
            temp_coeffs_j.append(coeffs)
            knot_counts_j.append(knots.size)

        max_num_knots = max(knot_counts_j) if knot_counts_j else 0
        max_num_coeffs_intervals = max_num_knots - 1
        knot_padding_value = jnp.inf
        coeff_padding_value = jnp.nan

        # Stack values into jnp arrays
        self.stacked_knots = jnp.full((j, max_num_knots), knot_padding_value, dtype=jnp.float32)
        self.stacked_coeffs = jnp.full((j, 4, max_num_coeffs_intervals), coeff_padding_value, dtype=jnp.float32)
        self.knot_counts = jnp.array(knot_counts_j, dtype=jnp.int32)
        for i in range(j):
            k = temp_knots_j[i]
            c = temp_coeffs_j[i]
            count = knot_counts_j[i]
            self.stacked_knots = self.stacked_knots.at[i, :count].set(jnp.array(k))
            self.stacked_coeffs = self.stacked_coeffs.at[i, :, : (count - 1)].set(jnp.array(c))

        self.has_spline_data_flags = jnp.ones(j, dtype=bool)
        self.default_error_gain = jnp.ones(j, dtype=jnp.float32)
        self.action_noise = action_noise
        self.action_noise_type = action_noise_type
        self.torque_noise = torque_noise
        self.torque_noise_type = torque_noise_type

    def _eval_spline(self, x_val_sat: Array, knots: Array, coeffs: Array, knot_count: Array) -> Array:
        """Evaluate spline using SciPy format on potentially padded arrays (dynamic slice fix)."""
        search_result = jnp.searchsorted(knots, x_val_sat)
        idx = jnp.clip(search_result - 1, 0, knot_count - 2)
        dx = x_val_sat - knots[idx]
        spline_value = coeffs[0, idx] * dx**3 + coeffs[1, idx] * dx**2 + coeffs[2, idx] * dx + coeffs[3, idx]
        return spline_value

    def get_ctrl(self, action_j: Array, physics_data: PhysicsData, rng: PRNGKeyArray) -> Array:
        """Compute torque control using Feetech parameters and mandatory cubic spline (JAX friendly)."""
        pos_rng, tor_rng = jax.random.split(rng)
        current_pos_j = physics_data.qpos[7:]
        current_vel_j = physics_data.qvel[6:]
        pos_error_j = action_j - current_pos_j
        vel_error_j = -current_vel_j

        def process_single_joint(err: Array, k: Array, c: Array, kc: Array, flag: Array, default_eg: Array) -> Array:
            abs_err = jnp.abs(err)
            x_clamped = jnp.clip(abs_err, k[0], k[kc - 1])  # Use knot_count 'kc' for safe indexing

            def eval_spline_branch() -> Array:
                return self._eval_spline(x_clamped, k, c, kc)

            def default_gain_branch() -> Array:
                return jnp.array(default_eg)

            result = jax.lax.cond(flag, eval_spline_branch, default_gain_branch)
            return result

        error_gain_j = jax.vmap(process_single_joint)(
            pos_error_j,
            self.stacked_knots,
            self.stacked_coeffs,
            self.knot_counts,
            self.has_spline_data_flags,
            self.default_error_gain,
        )

        duty_j = self.kp_j * error_gain_j * pos_error_j + self.kd_j * vel_error_j
        torque_j = jnp.clip(
            self.add_noise(self.torque_noise, self.torque_noise_type, duty_j * self.max_torque_j, tor_rng),
            -self.max_torque_j,
            self.max_torque_j,
        )
        return torque_j

    def get_default_action(self, physics_data: PhysicsData) -> Array:
        return physics_data.qpos[7:]


ZbotModel = TypeVar("ZbotModel", bound=eqx.Module)


class ZbotTask(ksim.PPOTask[Config], Generic[Config, ZbotModel]):
    @property
    @abc.abstractmethod
    def get_input_shapes(self) -> list[tuple[int, ...]]:
        """Returns a list of shapes expected by the exported model's inference function."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_optimizer(self) -> optax.GradientTransformation:
        raise NotImplementedError()

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

        sts3215_params_dict = self._load_feetech_params()
        sts3215_params = sts3215_params_dict["sts3215"]
        sts3250_params = sts3215_params_dict["sts3250"]

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

    def _load_feetech_params(self) -> dict[str, FeetechParams]:
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
        return {
            "sts3215": params_3215,
            "sts3250": params_3250,
        }

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: dict[str, JointMetadataOutput] | None = None,
    ) -> ksim.Actuators:
        if metadata is not None:
            joint_names = sorted(metadata.keys())

            num_joints = len(joint_names)
            max_torque_j = jnp.zeros(num_joints)
            kp_j = jnp.zeros(num_joints)
            kd_j = jnp.zeros(num_joints)

            feetech_params_dict = self._load_feetech_params()
            sts3215_params = feetech_params_dict["sts3215"]
            sts3250_params = feetech_params_dict["sts3250"]
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
            error_gain_data_list_j: list[ErrorGainData] = []

            for i, joint_name in enumerate(joint_names):
                joint_meta = metadata[joint_name]
                if "_15" in joint_name:
                    max_torque_j = max_torque_j.at[i].set(sts3215_params["max_torque"])
                    error_gain_data_list_j.append(sts3215_params["error_gain_data"])
                elif "_50" in joint_name:
                    max_torque_j = max_torque_j.at[i].set(sts3250_params["max_torque"])
                    error_gain_data_list_j.append(sts3250_params["error_gain_data"])
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

                kp_j = kp_j.at[i].set(kp_val)
                kd_j = kd_j.at[i].set(kd_val)
        else:
            raise ValueError("Metadata is not available")
        return FeetechActuators(
            max_torque_j=max_torque_j,
            kp_j=kp_j,
            kd_j=kd_j,
            error_gain_data_j=error_gain_data_list_j,
            action_noise=0.0,
            action_noise_type="none",
            torque_noise=0.0,
            torque_noise_type="none",
        )

    @abc.abstractmethod
    def get_randomization(self, physics_model: ksim.PhysicsModel) -> list[ksim.Randomization]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_model(self, key: PRNGKeyArray) -> ZbotModel:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_initial_carry(self, rng: PRNGKeyArray) -> Array:
        raise NotImplementedError()

    @abc.abstractmethod
    def _run_actor(
        self,
        model: ZbotModel,
        observations: FrozenDict[str, Array],
        commands: FrozenDict[str, Array],
    ) -> distrax.Normal:
        raise NotImplementedError()

    @abc.abstractmethod
    def _run_critic(
        self,
        model: ZbotModel,
        observations: FrozenDict[str, Array],
        commands: FrozenDict[str, Array],
    ) -> Array:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_on_policy_log_probs(
        self,
        model: ZbotModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> Array:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_on_policy_values(
        self,
        model: ZbotModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> Array:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_log_probs(
        self,
        model: ZbotModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> tuple[Array, Array]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_values(
        self,
        model: ZbotModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> Array:
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_action(
        self,
        model: ZbotModel,
        carry: Array,
        physics_model: ksim.PhysicsModel,
        physics_state: PhysicsState,
        observations: FrozenDict[str, Array],
        commands: FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> tuple[Array, Array, AuxOutputs]:
        raise NotImplementedError()

    def make_export_model(self, model: ZbotModel, stochastic: bool = False, batched: bool = False) -> Callable:
        """Makes a callable inference function that directly takes a flattened input vector and returns an action.

        Returns:
            A tuple containing the inference function and the size of the input vector.
        """
        # Cast model to the expected type to satisfy MyPy
        if not hasattr(model, "actor") or not hasattr(model.actor, "call_flat_obs"):
            raise TypeError("Model passed to make_export_model must have actor with call_flat_obs method.")

        def deterministic_model_fn(obs: Array) -> Array:
            # Use the cast model
            return model.actor.call_flat_obs(obs).mode()

        def stochastic_model_fn(obs: Array) -> Array:
            # Use the cast model
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

        input_shapes: list[tuple[int, ...]] = self.get_input_shapes

        export(
            model_fn,
            input_shapes,
            ckpt_path.parent / "tf_model",
        )

        return state
