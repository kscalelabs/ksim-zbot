"""Common definitions and utilities for Z-Bot tasks."""

import abc
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Optional, Sequence, TypedDict, TypeVar, Union, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import JointMetadataOutput
from ksim.actuators import NoiseType, StatefulActuators
from ksim.types import PhysicsData
from mujoco import mjx
from mujoco_scenes.mjcf import load_mjmodel

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AuxOutputs:
    log_probs: Array
    values: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PlannerState:
    position: Array
    velocity: Array


@dataclass
class ZbotTaskConfig(ksim.PPOConfig):
    """Config for the Z-Bot walking task."""

    robot_urdf_path: str = xax.field(
        value="ksim_zbot/kscale-assets/zbot-6dof-feet/",
        help="The path to the assets directory for the robot.",
    )

    actuator_params_path: str = xax.field(
        value="ksim_zbot/kscale-assets/actuators/",
        help="The path to the assets directory for actuator models",
    )

    domain_randomize: bool = xax.field(
        value=True,
        help="Whether to randomize the task parameters.",
    )

    # Checkpointing parameters.
    export_for_inference: bool = xax.field(
        value=True,
        help="Whether to export the model for inference.",
    )

    # Rendering parameters.
    render_track_body_id: int | None = xax.field(
        value=0,
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
    vmax: float
    amax: float
    max_velocity: float
    max_pwm: float
    error_gain: float


def trapezoidal_step(
    state: PlannerState, target_position: Array, dt: float
) -> tuple[PlannerState, tuple[Array, Array]]:
    v_max = 5.0
    a_max = 39.0

    position_error = target_position - state.position
    direction = jnp.sign(position_error)

    stopping_distance = (state.velocity**2) / (2 * a_max)

    # Decide accelerate or decelerate
    should_accelerate = jnp.abs(position_error) > stopping_distance

    acceleration = jnp.where(should_accelerate, direction * a_max, -direction * a_max)
    new_velocity = state.velocity + acceleration * dt

    # Clamp velocity
    new_velocity = jnp.clip(new_velocity, -v_max, v_max)

    # Prevent overshoot when decelerating
    new_velocity = jnp.where(direction * new_velocity < 0, 0.0, new_velocity)

    new_position = state.position + new_velocity * dt

    new_state = PlannerState(position=new_position, velocity=new_velocity)

    return new_state, (new_position, new_velocity)


def load_actuator_params(params_path: str, actuator_type: str) -> FeetechParams:
    params_file = Path(params_path) / f"{actuator_type}.json"
    if not params_file.exists():
        raise ValueError(
            f"Actuator parameters file '{params_file}' not found. Please ensure it exists in '{params_path}'."
        )
    with open(params_file, "r") as f:
        return json.load(f)


class FeetechActuators(StatefulActuators):
    """Feetech actuator controller."""

    def __init__(
        self,
        max_torque_j: Array,
        kp_j: Array,
        kd_j: Array,
        max_velocity_j: Array,
        max_pwm_j: Array,
        vin_j: Array,
        kt_j: Array,
        r_j: Array,
        vmax_j: Array,
        amax_j: Array,
        error_gain_j: Array,
        dt: float,
        action_noise: float = 0.0,
        action_noise_type: NoiseType = "none",
        torque_noise: float = 0.0,
        torque_noise_type: NoiseType = "none",
    ) -> None:
        self.max_torque_j = max_torque_j
        self.kp_j = kp_j
        self.kd_j = kd_j
        self.max_velocity_j = max_velocity_j
        self.max_pwm_j = max_pwm_j
        self.vin_j = vin_j
        self.kt_j = kt_j
        self.r_j = r_j
        self.vmax_j = vmax_j
        self.amax_j = amax_j
        self.error_gain_j = error_gain_j
        self.dt = dt
        # self.prev_qtarget_j = jnp.zeros_like(self.kp_j)
        self.action_noise = action_noise
        self.action_noise_type = action_noise_type
        self.torque_noise = torque_noise
        self.torque_noise_type = torque_noise_type

    def get_stateful_ctrl(
        self,
        action: Array,
        physics_data: PhysicsData,
        actuator_state: PlannerState,
        rng: PRNGKeyArray,
    ) -> tuple[Array, PlannerState]:
        """Compute torque control with velocity smoothing and duty cycle clipping (JAX friendly)."""
        pos_rng, tor_rng = jax.random.split(rng)

        current_pos_j = physics_data.qpos[7:]
        current_vel_j = physics_data.qvel[6:]

        actuator_state, (desired_position, desired_velocity) = trapezoidal_step(actuator_state, action, self.dt)

        pos_error_j = desired_position - current_pos_j
        vel_error_j = desired_velocity - current_vel_j

        # Compute raw duty cycle and clip by max_pwm
        raw_duty_j = self.kp_j * self.error_gain_j * pos_error_j + self.kd_j * vel_error_j
        duty_j = jnp.clip(raw_duty_j, -self.max_pwm_j, self.max_pwm_j)

        # Compute torque
        volts_j = duty_j * self.vin_j
        torque_j = volts_j * self.kt_j / self.r_j

        # Add noise to torque
        torque_j_noisy = self.add_noise(self.torque_noise, self.torque_noise_type, torque_j, tor_rng)

        return torque_j_noisy, actuator_state

    def get_default_action(self, physics_data: PhysicsData) -> Array:
        return physics_data.qpos[7:]

    def get_default_state(self, initial_position: Array, initial_velocity: Array) -> PlannerState:
        """Initialize the planner state with the provided position and velocity."""
        return PlannerState(position=initial_position, velocity=initial_velocity)

    def get_initial_state(self, physics_data: PhysicsData, rng: PRNGKeyArray) -> PlannerState:
        """Implement abstract method to initialize planner state from physics data."""
        initial_position = physics_data.qpos[7:]
        initial_velocity = physics_data.qvel[6:]
        return self.get_default_state(initial_position, initial_velocity)


ZbotModel = TypeVar("ZbotModel", bound=eqx.Module)


class ZbotTask(ksim.PPOTask[Config], Generic[Config, ZbotModel]):
    @property
    @abc.abstractmethod
    def get_input_shapes(self) -> list[tuple[int, ...]]:
        """Returns a list of shapes expected by the exported model's inference function."""
        raise NotImplementedError()

    def _configure_actuator_params(
        self,
        mj_model: mujoco.MjModel,
        dof_id: int,
        joint_name: str,
        params: FeetechParams,
    ) -> None:
        """Configure actuator parameters for a joint."""
        mj_model.dof_damping[dof_id] = params["damping"]
        mj_model.dof_armature[dof_id] = params["armature"]
        mj_model.dof_frictionloss[dof_id] = params["frictionloss"]
        actuator_name = f"{joint_name}_ctrl"
        actuator_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id >= 0:
            max_torque = float(params["max_torque"])
            mj_model.actuator_forcerange[actuator_id, :] = [-max_torque, max_torque]

    def get_mujoco_model(self) -> mujoco.MjModel:
        mjcf_path = (Path(self.config.robot_urdf_path) / "robot.mjcf").resolve().as_posix()
        logger.info("Loading MJCF model from %s", mjcf_path)
        mj_model = load_mjmodel(mjcf_path, scene="smooth")

        metadata = self.get_mujoco_model_metadata(mj_model)

        mj_model.opt.timestep = jnp.array(self.config.dt)
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        mj_model.opt.disableflags = mjx.DisableBit.EULERDAMP
        mj_model.opt.solver = mjx.SolverType.CG

        # Validate parameters
        required_keys = ["damping", "armature", "frictionloss", "max_torque"]

        # Apply servo-specific parameters based on joint metadata
        for i in range(mj_model.njnt):
            joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name is None:
                logger.warning("Joint at index %d has no name; skipping parameter assignment.", i)
                continue

            # Look up joint metadata. Warn if actuator_type is missing.
            if joint_name not in metadata:
                logger.warning("Joint '%s' is missing; skipping parameter assignment.", joint_name)
                continue

            joint_meta = metadata[joint_name]
            if joint_meta.actuator_type is None:
                logger.warning("Joint '%s' is missing an actuator_type; skipping parameter assignment.", joint_name)
                continue

            dof_id = mj_model.jnt_dofadr[i]

            # Load and validate parameters for this actuator type
            params = load_actuator_params(self.config.actuator_params_path, joint_meta.actuator_type)
            for key in required_keys:
                if key not in params:
                    raise ValueError(f"Missing required key '{key}' in {joint_meta.actuator_type} parameters.")

            # Apply parameters based on the joint suffix
            self._configure_actuator_params(mj_model, dof_id, joint_name, params)

        return mj_model

    def log_joint_config(self, model: Union[mujoco.MjModel, mjx.Model]) -> None:
        metadata = self.get_mujoco_model_metadata(model)
        debug_lines = ["==== Joint and Actuator Properties ===="]

        if isinstance(model, mujoco.MjModel):
            logger.info("******** PhysicsModel is Mujoco")

            njnt = model.njnt

            def get_joint_name(idx: int) -> Optional[str]:
                return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, idx)

            def get_actuator_id(name: str) -> int:
                return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

            dof_damping = model.dof_damping
            dof_armature = model.dof_armature
            dof_frictionloss = model.dof_frictionloss
            jnt_dofadr = model.jnt_dofadr
            actuator_forcerange = model.actuator_forcerange

        elif isinstance(model, mjx.Model):
            logger.info("******** PhysicsModel is MJX")

            njnt = model.njnt
            dof_damping = model.dof_damping
            dof_armature = model.dof_armature
            dof_frictionloss = model.dof_frictionloss
            jnt_dofadr = model.jnt_dofadr
            actuator_forcerange = model.actuator_forcerange

            def extract_name(byte_array: bytes, adr_array: Sequence[int], idx: int) -> Optional[str]:
                adr = adr_array[idx]
                if adr < 0:
                    return None
                end = byte_array.find(b"\x00", adr)
                return byte_array[adr:end].decode("utf-8")

            actuator_name_to_id = {
                extract_name(model.names, model.name_actuatoradr, i): i
                for i in range(model.nu)
                if model.name_actuatoradr[i] >= 0
            }

            def get_joint_name(idx: int) -> Optional[str]:
                return extract_name(model.names, model.name_jntadr, idx)

            def get_actuator_id(name: str) -> int:
                return actuator_name_to_id.get(name, -1)

        else:
            raise TypeError("Unsupported model type provided")

        for i in range(njnt):
            joint_name = get_joint_name(i)
            if joint_name is None:
                continue

            joint_meta = metadata.get(joint_name)
            if not joint_meta:
                logger.warning("Joint '%s' missing metadata; skipping.", joint_name)
                continue

            actuator_type = joint_meta.actuator_type
            if actuator_type is None:
                logger.warning("Joint '%s' missing actuator_type; skipping.", joint_name)
                continue

            dof_id = jnt_dofadr[i]
            damping = dof_damping[dof_id]
            armature = dof_armature[dof_id]
            frictionloss = dof_frictionloss[dof_id]
            joint_id = joint_meta.id if joint_meta.id is not None else "N/A"
            kp = joint_meta.kp if joint_meta.kp is not None else "N/A"
            kd = joint_meta.kd if joint_meta.kd is not None else "N/A"

            actuator_name = f"{joint_name}_ctrl"
            actuator_id = get_actuator_id(actuator_name)

            line = (
                f"Joint: {joint_name:<20} | Joint ID: {joint_id!s:<3} | "
                f"Damping: {damping:6.3f} | Armature: {armature:6.3f} | "
                f"Friction: {frictionloss:6.3f}"
            )

            if actuator_id >= 0:
                forcerange = actuator_forcerange[actuator_id]
                line += (
                    f" | Actuator: {actuator_name:<20} (ID: {actuator_id:2d}) | "
                    f"Forcerange: [{forcerange[0]:6.3f}, {forcerange[1]:6.3f}] | "
                    f"Kp: {kp} | Kd: {kd}"
                )
            else:
                line += " | Actuator: N/A (passive joint)"

            debug_lines.append(line)

        logger.info("\n".join(debug_lines))

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> dict[str, JointMetadataOutput]:
        """Get joint metadata from metadata.json file."""
        metadata_path = Path(self.config.robot_urdf_path) / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        try:
            with open(metadata_path, "r") as f:
                raw_metadata = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {metadata_path}: {e}")

        if "joint_name_to_metadata" not in raw_metadata:
            raise ValueError(f"'joint_name_to_metadata' key missing in {metadata_path}")

        joint_metadata = raw_metadata["joint_name_to_metadata"]
        if not isinstance(joint_metadata, dict):
            raise TypeError(f"'joint_name_to_metadata' in {metadata_path} must be a dictionary.")

        # Convert raw metadata to JointMetadataOutput objects
        return {joint_name: JointMetadataOutput(**metadata) for joint_name, metadata in joint_metadata.items()}

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: dict[str, JointMetadataOutput] | None = None,
    ) -> ksim.Actuators:
        if metadata is None:
            raise ValueError("Metadata is required to get actuators")

        self.joint_mappings = self.create_joint_mappings(physics_model, metadata)

        num_joints = len(self.joint_mappings)

        max_torque_j = jnp.zeros(num_joints)
        max_velocity_j = jnp.zeros(num_joints)
        max_pwm_j = jnp.zeros(num_joints)
        vin_j = jnp.zeros(num_joints)
        kt_j = jnp.zeros(num_joints)
        r_j = jnp.zeros(num_joints)
        # vmax_j = jnp.zeros(num_joints)
        # amax_j = jnp.zeros(num_joints)
        vmax_j = jnp.ones(num_joints) * 5.0  # or whatever test value you want
        amax_j = jnp.ones(num_joints) * 39.0
        kp_j = jnp.zeros(num_joints)
        kd_j = jnp.zeros(num_joints)
        error_gain_j = jnp.zeros(num_joints)

        # Validate parameters
        required_keys = ["max_torque", "error_gain", "max_velocity", "max_pwm", "vin", "kt", "R"]

        # Sort joint_mappings by actuator_id to ensure correct ordering
        sorted_joints = sorted(self.joint_mappings.items(), key=lambda x: x[1]["actuator_id"])

        for i, (joint_name, mapping) in enumerate(sorted_joints):
            joint_metadata = metadata[joint_name]
            if not isinstance(joint_metadata, JointMetadataOutput):
                raise TypeError(f"Metadata entry for joint '{joint_name}' must be a JointMetadataOutput.")

            actuator_type = cast(str, joint_metadata.actuator_type)
            if actuator_type is None:
                raise ValueError(f"'actuator_type' is not available for joint {joint_name}")
            if not isinstance(actuator_type, str):
                raise TypeError(f"'actuator_type' for joint {joint_name} must be a string.")

            params = load_actuator_params(self.config.actuator_params_path, actuator_type)

            # Validate parameters
            for key in required_keys:
                if key not in params:
                    raise ValueError(f"Missing required key '{key}' in {actuator_type} parameters.")

            max_torque_j = max_torque_j.at[i].set(params["max_torque"])
            max_velocity_j = max_velocity_j.at[i].set(params["max_velocity"])
            max_pwm_j = max_pwm_j.at[i].set(params["max_pwm"])
            vin_j = vin_j.at[i].set(params["vin"])
            kt_j = kt_j.at[i].set(params["kt"])
            r_j = r_j.at[i].set(params["R"])
            error_gain_j = error_gain_j.at[i].set(params["error_gain"])

            # Set kp and kd values
            if joint_metadata.kp is None or joint_metadata.kd is None:
                raise ValueError(f"kp/kd values for joint {joint_name} are not available")
            kp_j = kp_j.at[i].set(float(joint_metadata.kp))
            kd_j = kd_j.at[i].set(float(joint_metadata.kd))

        self.log_joint_config(physics_model)

        return FeetechActuators(
            max_torque_j=max_torque_j,
            max_velocity_j=max_velocity_j,
            max_pwm_j=max_pwm_j,
            vin_j=vin_j,
            kt_j=kt_j,
            r_j=r_j,
            kp_j=kp_j,
            kd_j=kd_j,
            vmax_j=vmax_j,
            amax_j=amax_j,
            dt=self.config.dt,
            error_gain_j=error_gain_j,
            action_noise=0.0,
            action_noise_type="none",
            torque_noise=0.0,
            torque_noise_type="none",
        )

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

    def on_after_checkpoint_save(self, ckpt_path: Path, state: xax.State | None) -> xax.State | None:
        if state is None:
            return None

        state = super().on_after_checkpoint_save(ckpt_path, state)

        # model: ZbotModel = self.load_ckpt(ckpt_path, part="model")

        # model_fn = self.make_export_model(model, stochastic=False, batched=True)

        # input_shapes: list[tuple[int, ...]] = self.get_input_shapes

        # export(
        #     model_fn,
        #     input_shapes,
        #     ckpt_path.parent / "tf_model",
        # )

        return state

    def create_joint_mappings(
        self, physics_model: ksim.PhysicsModel, metadata: dict[str, JointMetadataOutput]
    ) -> dict[str, dict]:
        """Creates mappings between joint names, nn_ids, and actuator_ids.

        Args:
            physics_model: The MuJoCo/MJX model containing joint information
            metadata: The joint metadata dictionary from metadata.json

        Returns:
            Dictionary mapping joint names to their nn_id and actuator_id
        """
        debug_lines = ["==== Joint Name to ID Mappings ===="]

        # Get ordered list of joints from MuJoCo/MJX model
        if isinstance(physics_model, mujoco.MjModel):
            mujoco_joints = [
                mujoco.mj_id2name(physics_model, mujoco.mjtObj.mjOBJ_JOINT, i)
                for i in range(physics_model.njnt)
                if mujoco.mj_id2name(physics_model, mujoco.mjtObj.mjOBJ_JOINT, i) is not None
            ]
        else:  # MJX model

            def extract_joint_name(model: mjx.Model, idx: int) -> Optional[str]:
                adr = model.name_jntadr[idx]
                if adr < 0:
                    return None
                end = model.names.find(b"\x00", adr)
                return model.names[adr:end].decode("utf-8")

            mujoco_joints = [
                name for i in range(physics_model.njnt) if (name := extract_joint_name(physics_model, i)) is not None
            ]

        # Create mappings using joint names as keys
        joint_mappings = {}

        # Map each joint, using MuJoCo order for nn_ids
        for nn_id, joint_name in enumerate(mujoco_joints):
            if joint_name in metadata:
                actuator_id = metadata[joint_name].id
                if actuator_id is None:
                    logger.warning("Joint %s has no actuator id", joint_name)
                joint_mappings[joint_name] = {"nn_id": nn_id, "actuator_id": actuator_id}

                debug_lines.append("%-30s -> nn_id: %2d, actuator_id: %s" % (joint_name, nn_id, str(actuator_id)))
            else:
                logger.warning("Joint %s not found in metadata", joint_name)

        logger.info("\n".join(debug_lines))
        return joint_mappings
