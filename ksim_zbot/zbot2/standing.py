"""Defines simple task for training a walking policy for Z-Bot."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, TypeVar

import attrs
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import mujoco
import optax
import xax
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import JointMetadataOutput
from mujoco import mjx
from mujoco_scenes.mjcf import load_mjmodel
from xax.nn.export import export
from ksim.types import PhysicsData
from ksim.actuators import Actuators, NoiseType

OBS_SIZE = 20 * 2 + 3 + 3 + 20  # position + velocity + imu_acc + imu_gyro + last_action
CMD_SIZE = 2
NUM_OUTPUTS = 20  # position only for FeetechActuators (not position + velocity)

SINGLE_STEP_HISTORY_SIZE = NUM_OUTPUTS + OBS_SIZE + CMD_SIZE

HISTORY_LENGTH = 0

NUM_INPUTS = (OBS_SIZE + CMD_SIZE) + SINGLE_STEP_HISTORY_SIZE * HISTORY_LENGTH

# Feetech parameters from Scott's modelling
FT_STS3215_PARAMS = {
    "sysid": "sts3215-12v-id009",  # Traceable ID @ github.com/kscalelabs/sysid
    "max_torque": 5.466091040935576,
    "armature": 0.039999999991812,
    "frictionloss": 0.11434146818509992,
    "damping": 1.2305092028680242,
    "vin": 12.1,
    "kt": 1.0000000244338463,
    "R": 2.2136477795617733,
    "error_gain_data": [
        {"pos_err": 0.003067962, "error_gain": 0.257703684},
        {"pos_err": 0.004601942, "error_gain": 0.226127343},
        {"pos_err": 0.006135923, "error_gain": 0.210339173},
        {"pos_err": 0.007669904, "error_gain": 0.200866271},
        {"pos_err": 0.009203885, "error_gain": 0.194551002},
        {"pos_err": 0.010737866, "error_gain": 0.190040097},
        {"pos_err": 0.012271846, "error_gain": 0.186656917},
        {"pos_err": 0.013805827, "error_gain": 0.184025556},
        {"pos_err": 0.015339808, "error_gain": 0.181920466},
        {"pos_err": 0.016873789, "error_gain": 0.18019812},
        {"pos_err": 0.018407769, "error_gain": 0.178762832},
        {"pos_err": 0.01994175, "error_gain": 0.177548357},
        {"pos_err": 0.021475731, "error_gain": 0.176507379},
        {"pos_err": 0.023009712, "error_gain": 0.175605198},
        {"pos_err": 0.024543693, "error_gain": 0.174815789},
        {"pos_err": 0.026077673, "error_gain": 0.174119253},
        {"pos_err": 0.027611654, "error_gain": 0.173500109},
        {"pos_err": 0.029145635, "error_gain": 0.172838918},
        {"pos_err": 0.030679616, "error_gain": 0.172447564},
        {"pos_err": 0.032213597, "error_gain": 0.171996473},
        {"pos_err": 0.033747577, "error_gain": 0.171586391},
        {"pos_err": 0.035281558, "error_gain": 0.171211968},
        {"pos_err": 0.036815539, "error_gain": 0.170783864},
        {"pos_err": 0.03834952, "error_gain": 0.17051224},
        {"pos_err": 0.0398835, "error_gain": 0.170222333},
        {"pos_err": 0.041417481, "error_gain": 0.169953901},
        {"pos_err": 0.042951462, "error_gain": 0.169704642},
        {"pos_err": 0.044485443, "error_gain": 0.169472574},
        {"pos_err": 0.046019424, "error_gain": 0.169255977},
        {"pos_err": 0.047553404, "error_gain": 0.169053354},
        {"pos_err": 0.049087385, "error_gain": 0.168863395},
        {"pos_err": 0.050621366, "error_gain": 0.168684948},
        {"pos_err": 0.052155347, "error_gain": 0.168516999},
        {"pos_err": 0.053689328, "error_gain": 0.168358646},
        {"pos_err": 0.055223308, "error_gain": 0.168209091},
        {"pos_err": 0.056757289, "error_gain": 0.16806762},
        {"pos_err": 0.05829127, "error_gain": 0.167922873},
        {"pos_err": 0.059825251, "error_gain": 0.167795995},
        {"pos_err": 0.061359232, "error_gain": 0.167675462},
        {"pos_err": 0.062893212, "error_gain": 0.167560808},
        {"pos_err": 0.064427193, "error_gain": 0.167451614},
        {"pos_err": 0.065961174, "error_gain": 0.167347499},
        {"pos_err": 0.067495155, "error_gain": 0.167248117},
        {"pos_err": 0.069029135, "error_gain": 0.167153151},
        {"pos_err": 0.070563116, "error_gain": 0.167049028},
        {"pos_err": 0.072097097, "error_gain": 0.16696234},
        {"pos_err": 0.073631078, "error_gain": 0.166879263},
        {"pos_err": 0.075165059, "error_gain": 0.166799577},
        {"pos_err": 0.076699039, "error_gain": 0.166723079},
        {"pos_err": 0.07823302, "error_gain": 0.166649581},
        {"pos_err": 0.079767001, "error_gain": 0.166578909},
        {"pos_err": 0.081300982, "error_gain": 0.166510904},
        {"pos_err": 0.082834963, "error_gain": 0.166445418},
        {"pos_err": 0.084368943, "error_gain": 0.166382314},
        {"pos_err": 0.085902924, "error_gain": 0.166303274},
        {"pos_err": 0.087436905, "error_gain": 0.166244877},
        {"pos_err": 0.088970886, "error_gain": 0.166199031},
        {"pos_err": 0.090504866, "error_gain": 0.166130569},
        {"pos_err": 0.092038847, "error_gain": 0.166081366},
        {"pos_err": 0.093572828, "error_gain": 0.166030437},
        {"pos_err": 0.095106809, "error_gain": 0.16598115},
        {"pos_err": 0.09664079, "error_gain": 0.165933428},
        {"pos_err": 0.09817477, "error_gain": 0.165887197},
        {"pos_err": 0.099708751, "error_gain": 0.165842389},
        {"pos_err": 0.101242732, "error_gain": 0.165798939},
        {"pos_err": 0.102776713, "error_gain": 0.165750704},
        {"pos_err": 0.104310694, "error_gain": 0.16570988},
        {"pos_err": 0.105844674, "error_gain": 0.165667287},
        {"pos_err": 0.107378655, "error_gain": 0.165628821},
        {"pos_err": 0.108912636, "error_gain": 0.165585699},
        {"pos_err": 0.110446617, "error_gain": 0.165549435},
        {"pos_err": 0.111980598, "error_gain": 0.165514164},
        {"pos_err": 0.113514578, "error_gain": 0.165479847},
        {"pos_err": 0.115048559, "error_gain": 0.165446444},
        {"pos_err": 0.11658254, "error_gain": 0.165413921},
        {"pos_err": 0.118116521, "error_gain": 0.165382242},
        {"pos_err": 0.119650501, "error_gain": 0.165351376},
        {"pos_err": 0.121184482, "error_gain": 0.165321291},
        {"pos_err": 0.122718463, "error_gain": 0.165291958},
        {"pos_err": 0.124252444, "error_gain": 0.165263349},
        {"pos_err": 0.125786425, "error_gain": 0.165235438},
        {"pos_err": 0.127320405, "error_gain": 0.1652082},
        {"pos_err": 0.128854386, "error_gain": 0.16518161},
        {"pos_err": 0.130388367, "error_gain": 0.165155646},
        {"pos_err": 0.131922348, "error_gain": 0.165130286},
        {"pos_err": 0.133456329, "error_gain": 0.165082093},
        {"pos_err": 0.134990309, "error_gain": 0.165058145},
        {"pos_err": 0.13652429, "error_gain": 0.165034735},
        {"pos_err": 0.138058271, "error_gain": 0.165011845},
        {"pos_err": 0.139592252, "error_gain": 0.164989458},
        {"pos_err": 0.141126232, "error_gain": 0.164967558},
        {"pos_err": 0.142660213, "error_gain": 0.164946129},
        {"pos_err": 0.144194194, "error_gain": 0.164925156},
        {"pos_err": 0.145728175, "error_gain": 0.164904625},
        {"pos_err": 0.147262156, "error_gain": 0.164873911},
        {"pos_err": 0.148796136, "error_gain": 0.164854331},
        {"pos_err": 0.150330117, "error_gain": 0.164835151},
        {"pos_err": 0.151864098, "error_gain": 0.164816358},
        {"pos_err": 0.153398079, "error_gain": 0.164787755},
    ],
    "error_gain": 0.164787755,
}


FT_STS3250_PARAMS = {
    "sysid": "sts3250-id008",  # Traceable ID @ github.com/kscalelabs/sysid
    "max_torque": 8.716130441407099,
    "armature": 0.03999977737144798,
    "damping": 1.3464038511725651,
    "frictionloss": 0.19999504581400715,
    "vin": 12.1,
    "kt": 1.0005874626213263,
    "R": 1.3890462492623645,
    "error_gain_data": [
        {"pos_err": 0.001533981, "error_gain": 0.198625369},
        {"pos_err": 0.003067962, "error_gain": 0.180800015},
        {"pos_err": 0.004601942, "error_gain": 0.175129855},
        {"pos_err": 0.006135923, "error_gain": 0.172091057},
        {"pos_err": 0.007669904, "error_gain": 0.170267778},
        {"pos_err": 0.009203885, "error_gain": 0.168984352},
        {"pos_err": 0.010737866, "error_gain": 0.168125825},
        {"pos_err": 0.012271846, "error_gain": 0.167507395},
        {"pos_err": 0.013805827, "error_gain": 0.167003758},
        {"pos_err": 0.015339808, "error_gain": 0.166539732},
        {"pos_err": 0.016873789, "error_gain": 0.166215635},
        {"pos_err": 0.018407769, "error_gain": 0.165962531},
        {"pos_err": 0.01994175, "error_gain": 0.165748365},
        {"pos_err": 0.021475731, "error_gain": 0.165550243},
        {"pos_err": 0.023009712, "error_gain": 0.165378538},
        {"pos_err": 0.024543693, "error_gain": 0.165202831},
        {"pos_err": 0.026077673, "error_gain": 0.165071762},
        {"pos_err": 0.027611654, "error_gain": 0.164955257},
        {"pos_err": 0.029145635, "error_gain": 0.164851015},
        {"pos_err": 0.030679616, "error_gain": 0.164757197},
        {"pos_err": 0.032213597, "error_gain": 0.164672314},
        {"pos_err": 0.033747577, "error_gain": 0.164595148},
        {"pos_err": 0.035281558, "error_gain": 0.164524692},
        {"pos_err": 0.036815539, "error_gain": 0.164460108},
        {"pos_err": 0.03834952, "error_gain": 0.164376244},
        {"pos_err": 0.0398835, "error_gain": 0.164330172},
        {"pos_err": 0.041417481, "error_gain": 0.164272423},
        {"pos_err": 0.042951462, "error_gain": 0.164226074},
        {"pos_err": 0.044485443, "error_gain": 0.164168873},
        {"pos_err": 0.046019424, "error_gain": 0.164142647},
        {"pos_err": 0.047553404, "error_gain": 0.164091827},
        {"pos_err": 0.049087385, "error_gain": 0.164056915},
        {"pos_err": 0.050621366, "error_gain": 0.16402412},
        {"pos_err": 0.052155347, "error_gain": 0.163993253},
        {"pos_err": 0.053689328, "error_gain": 0.163964151},
        {"pos_err": 0.055223308, "error_gain": 0.163936665},
        {"pos_err": 0.056757289, "error_gain": 0.163910665},
        {"pos_err": 0.05829127, "error_gain": 0.163859228},
        {"pos_err": 0.059825251, "error_gain": 0.1638261},
        {"pos_err": 0.061359232, "error_gain": 0.163840465},
        {"pos_err": 0.062893212, "error_gain": 0.16380941},
        {"pos_err": 0.064427193, "error_gain": 0.163784685},
        {"pos_err": 0.065961174, "error_gain": 0.163770584},
        {"pos_err": 0.067495155, "error_gain": 0.163715456},
        {"pos_err": 0.069029135, "error_gain": 0.16373521},
        {"pos_err": 0.070563116, "error_gain": 0.163705391},
        {"pos_err": 0.072097097, "error_gain": 0.163689843},
        {"pos_err": 0.073631078, "error_gain": 0.163674943},
        {"pos_err": 0.075165059, "error_gain": 0.163660652},
        {"pos_err": 0.076699039, "error_gain": 0.163646932},
        {"pos_err": 0.07823302, "error_gain": 0.16363375},
        {"pos_err": 0.079767001, "error_gain": 0.163621076},
        {"pos_err": 0.081300982, "error_gain": 0.163608879},
        {"pos_err": 0.082834963, "error_gain": 0.163589589},
        {"pos_err": 0.084368943, "error_gain": 0.163578409},
        {"pos_err": 0.085902924, "error_gain": 0.16356399},
        {"pos_err": 0.087436905, "error_gain": 0.163557225},
        {"pos_err": 0.088970886, "error_gain": 0.163543668},
        {"pos_err": 0.090504866, "error_gain": 0.163544382},
        {"pos_err": 0.092038847, "error_gain": 0.163528097},
        {"pos_err": 0.093572828, "error_gain": 0.163525703},
        {"pos_err": 0.095106809, "error_gain": 0.163516815},
        {"pos_err": 0.09664079, "error_gain": 0.163501742},
        {"pos_err": 0.09817477, "error_gain": 0.163483958},
        {"pos_err": 0.099708751, "error_gain": 0.163460452},
        {"pos_err": 0.101242732, "error_gain": 0.163437658},
        {"pos_err": 0.102776713, "error_gain": 0.163491559},
        {"pos_err": 0.104310694, "error_gain": 0.163453999},
        {"pos_err": 0.105844674, "error_gain": 0.163447052},
        {"pos_err": 0.107378655, "error_gain": 0.163434483},
        {"pos_err": 0.108912636, "error_gain": 0.163419399},
        {"pos_err": 0.110446617, "error_gain": 0.163399075},
        {"pos_err": 0.111980598, "error_gain": 0.163393261},
        {"pos_err": 0.113514578, "error_gain": 0.163401369},
        {"pos_err": 0.115048559, "error_gain": 0.163403828},
        {"pos_err": 0.11658254, "error_gain": 0.163376737},
        {"pos_err": 0.118116521, "error_gain": 0.163382098},
        {"pos_err": 0.119650501, "error_gain": 0.163379487},
        {"pos_err": 0.121184482, "error_gain": 0.163361469},
        {"pos_err": 0.122718463, "error_gain": 0.163356634},
        {"pos_err": 0.124252444, "error_gain": 0.163351918},
        {"pos_err": 0.125786425, "error_gain": 0.163347317},
        {"pos_err": 0.127320405, "error_gain": 0.163342827},
        {"pos_err": 0.128854386, "error_gain": 0.163338444},
        {"pos_err": 0.130388367, "error_gain": 0.163334165},
        {"pos_err": 0.131922348, "error_gain": 0.163329984},
        {"pos_err": 0.133456329, "error_gain": 0.163314192},
        {"pos_err": 0.134990309, "error_gain": 0.163321909},
        {"pos_err": 0.13652429, "error_gain": 0.163306562},
        {"pos_err": 0.138058271, "error_gain": 0.163302875},
        {"pos_err": 0.139592252, "error_gain": 0.163299268},
        {"pos_err": 0.141126232, "error_gain": 0.16329574},
        {"pos_err": 0.142660213, "error_gain": 0.163281334},
        {"pos_err": 0.144194194, "error_gain": 0.163284574},
        {"pos_err": 0.145728175, "error_gain": 0.163274878},
        {"pos_err": 0.147262156, "error_gain": 0.163292972},
        {"pos_err": 0.148796136, "error_gain": 0.163268688},
        {"pos_err": 0.150330117, "error_gain": 0.163265688},
        {"pos_err": 0.151864098, "error_gain": 0.163268922},
        {"pos_err": 0.153398079, "error_gain": 0.163249681},
    ],
    "error_gain": 0.163249681,
}

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


class FeetechActuators(Actuators):
    """Feetech actuator controller for feetech motors (3215 and 3250) with derivative (kd) term."""

    def __init__(
        self,
        max_torque: float,
        vin: float,
        kt: float,
        R: float,
        error_gain: float,
        action_noise: float = 0.0,
        action_noise_type: NoiseType = "none",
        torque_noise: float = 0.0,
        torque_noise_type: NoiseType = "none",
    ) -> None:
        self.max_torque = max_torque
        self.vin = vin
        self.kt = kt
        self.R = R
        self.kp = 32
        self.kd = 32
        self.error_gain = error_gain
        self.action_noise = action_noise
        self.action_noise_type = action_noise_type
        self.torque_noise = torque_noise
        self.torque_noise_type = torque_noise_type

    def get_ctrl(self, action: Array, physics_data: PhysicsData, rng: PRNGKeyArray) -> Array:
        """
        Compute torque control using Feetech parameters.
        Assumes `action` is the target position.
        """
        pos_rng, tor_rng = jax.random.split(rng)
        # Extract current joint positions and velocities (ignoring root if necessary)
        current_pos = physics_data.qpos[7:]
        current_vel = physics_data.qvel[6:]

        # Compute position error (target position minus current position)
        pos_error = action - current_pos
        # Assume target velocity is zero; compute velocity error
        vel_error = -current_vel

        # Compute the combined control (PD control law)
        duty = self.kp * self.error_gain * pos_error + self.kd * vel_error

        # Multiply by max torque, add torque noise, and clip to limits
        torque = jnp.clip(
            self.add_noise(self.torque_noise, self.torque_noise_type, duty * self.max_torque, tor_rng),
            -self.max_torque,
            self.max_torque,
        )
        return torque

    def get_default_action(self, physics_data: PhysicsData) -> Array:
        # Default action: current joint positions.
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

    healthy_z_lower: float = attrs.field(default=0.5)
    healthy_z_upper: float = attrs.field(default=1.5)

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

    # Mujoco parameters.
    # Removed use_mit_actuators config option since we're now using FeetechActuators directly

    # Rendering parameters.
    render_track_body_id: int | None = xax.field(
        value=None,
        help="The body id to track with the render camera.",
    )

    # Checkpointing parameters.
    export_for_inference: bool = xax.field(
        value=False,
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

        # Apply servo-specific parameters based on joint name suffix
        for i in range(mj_model.njnt):
            joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name is None or not any(suffix in joint_name for suffix in ["_15", "_50"]):
                continue

            dof_id = mj_model.jnt_dofadr[i]

            # Apply parameters based on the joint suffix
            if "_15" in joint_name:  # STS3215 servos (arms)
                mj_model.dof_damping[dof_id] = FT_STS3215_PARAMS["damping"]
                mj_model.dof_armature[dof_id] = FT_STS3215_PARAMS["armature"]
                mj_model.dof_frictionloss[dof_id] = FT_STS3215_PARAMS["frictionloss"]

                # Get base name for actuator (remove the _15 suffix)
                base_name = joint_name.rsplit("_", 1)[0]
                actuator_name = f"{base_name}_15_ctrl"

                actuator_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                if actuator_id >= 0:
                    mj_model.actuator_ctrlrange[actuator_id, :] = [
                        -FT_STS3215_PARAMS["max_torque"],
                        FT_STS3215_PARAMS["max_torque"],
                    ]

            elif "_50" in joint_name:  # STS3250 servos (legs)
                mj_model.dof_damping[dof_id] = FT_STS3250_PARAMS["damping"]
                mj_model.dof_armature[dof_id] = FT_STS3250_PARAMS["armature"]
                mj_model.dof_frictionloss[dof_id] = FT_STS3250_PARAMS["frictionloss"]

                # Get base name for actuator (remove the _50 suffix)
                base_name = joint_name.rsplit("_", 1)[0]
                actuator_name = f"{base_name}_50_ctrl"

                actuator_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                if actuator_id >= 0:
                    mj_model.actuator_ctrlrange[actuator_id, :] = [
                        -FT_STS3250_PARAMS["max_torque"],
                        FT_STS3250_PARAMS["max_torque"],
                    ]

        return mj_model

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> dict[str, JointMetadataOutput]:
        metadata = asyncio.run(ksim.get_mujoco_model_metadata(self.config.robot_urdf_path, cache=False))
        if metadata.joint_name_to_metadata is None:
            raise ValueError("Joint metadata is not available")
        return metadata.joint_name_to_metadata

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: dict[str, JointMetadataOutput] | None = None,
    ) -> ksim.Actuators:
        return FeetechActuators(
            max_torque=FT_STS3250_PARAMS["max_torque"],
            vin=FT_STS3250_PARAMS["vin"],
            kt=FT_STS3250_PARAMS["kt"],
            R=FT_STS3250_PARAMS["R"],
            error_gain=FT_STS3250_PARAMS["error_gain"],
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
            ksim.MassMultiplicationRandomization.from_body_name(physics_model, "Z_BOT2_MASTER_BODY_SKELETON"),
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
                    0.40,  # Lower height from 0.91 to 0.41
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
            ksim.SensorObservation.create(physics_model, "IMU_acc", noise=0.5),
            ksim.SensorObservation.create(physics_model, "IMU_gyro", noise=0.2),
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
        imu_acc_3 = observations["IMU_acc_obs"]
        imu_gyro_3 = observations["IMU_gyro_obs"]
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
        imu_acc_3 = observations["IMU_acc_obs"]
        imu_gyro_3 = observations["IMU_gyro_obs"]
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
        imu_acc_3 = observations["IMU_acc_obs"]
        imu_gyro_3 = observations["IMU_gyro_obs"]
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

        input_shapes = [(NUM_INPUTS,)]

        export(  # type: ignore[operator]
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
            # Removed use_mit_actuators since we're using FeetechActuators now
            export_for_inference=True,
        ),
    )
