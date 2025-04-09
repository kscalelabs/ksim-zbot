"""Example script to deploy a SavedModel in KOS-Sim."""

import argparse
import asyncio
import json
import logging
import signal
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pykos
import tensorflow as tf

logger = logging.getLogger(__name__)

DT = 0.02  # Policy time step (50Hz)


@dataclass
class Actuator:
    actuator_id: int
    nn_id: int
    kp: float
    kd: float
    max_torque: float
    joint_name: str


def load_actuator_mapping(metadata_path: str | Path) -> dict:
    """Load actuator mapping from metadata file."""
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    joint_metadata = metadata.get("joint_name_to_metadata", {})

    # Create mapping of actuator IDs to neural network IDs (sorted by actuator ID)
    actuator_mapping = {}
    for joint_name, info in joint_metadata.items():
        try:
            actuator_id = int(info["id"])
            actuator_mapping[actuator_id] = {
                "joint_name": joint_name,
                "nn_id": None,  # Will be filled based on sorted order
            }
        except (KeyError, ValueError) as e:
            logger.warning("Invalid actuator ID for joint %s: %s", joint_name, e)
            continue

    # Assign nn_ids based on sorted actuator IDs
    for nn_id, actuator_id in enumerate(sorted(actuator_mapping.keys())):
        actuator_mapping[actuator_id]["nn_id"] = nn_id

    return actuator_mapping


async def get_observation(
    kos: pykos.KOS, actuator_mapping: dict, prev_action: np.ndarray, cmd: np.ndarray = np.array([0.15, 0.0])
) -> np.ndarray:
    """Get observation using actuator mapping from metadata."""
    actuator_ids = list(actuator_mapping.keys())

    (actuator_states, imu) = await asyncio.gather(
        kos.actuator.get_actuators_state(actuator_ids),
        kos.imu.get_imu_values(),
    )

    # Create arrays sorted by nn_id
    sorted_by_nn = sorted(actuator_mapping.items(), key=lambda x: x[1]["nn_id"])

    # Build position and velocity observations
    state_dict_pos = {state.actuator_id: state.position for state in actuator_states.states}
    state_dict_vel = {state.actuator_id: state.velocity for state in actuator_states.states}

    pos_obs = np.deg2rad([state_dict_pos[act_id] for act_id, _ in sorted_by_nn])
    vel_obs = np.deg2rad([state_dict_vel[act_id] for act_id, _ in sorted_by_nn])

    imu_obs = np.array([imu.accel_x, imu.accel_y, imu.accel_z, imu.gyro_x, imu.gyro_y, imu.gyro_z])

    last_action = prev_action  # Add last action to observation

    observation = np.concatenate([pos_obs, vel_obs, imu_obs, cmd, last_action], axis=-1)
    return observation


async def send_actions(kos: pykos.KOS, position: np.ndarray, actuator_mapping: dict) -> None:
    """Send actions using actuator mapping from metadata."""
    position = np.rad2deg(position)

    # Create commands sorted by nn_id to match position array
    sorted_by_nn = sorted(actuator_mapping.items(), key=lambda x: x[1]["nn_id"])
    actuator_commands: list[pykos.services.actuator.ActuatorCommand] = [
        {
            "actuator_id": actuator_id,
            "position": position[mapping["nn_id"]],
        }
        for actuator_id, mapping in sorted_by_nn
    ]

    logger.debug(actuator_commands)
    await kos.actuator.command_actuators(actuator_commands)


def load_feetech_params(actuator_params_path: str) -> tuple[dict, dict]:
    """Load Feetech parameters from files."""
    params_path = Path(actuator_params_path)
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
        params_3215 = json.load(f)
    with open(params_file_3250, "r") as f:
        params_3250 = json.load(f)
    return params_3215, params_3250


async def configure_actuators(
    kos: pykos.KOS, robot_urdf_path: str, actuator_params_path: str, metadata_path: str | None = None
) -> None:
    """Configure actuators using parameters from files."""
    # Load the Feetech parameters
    sts3215_params, sts3250_params = load_feetech_params(actuator_params_path)

    if metadata_path:
        metadata_file = Path(metadata_path)
    else:
        metadata_file = Path(robot_urdf_path) / "metadata.json"

    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_file}")

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    joint_metadata = metadata.get("joint_name_to_metadata", {})

    # Configure each actuator from metadata
    for joint_name, joint_info in joint_metadata.items():
        try:
            actuator_id = int(joint_info["id"])
            actuator_type = joint_info.get("actuator_type", "")
            kp = float(joint_info["kp"])
            kd = float(joint_info["kd"])
        except (KeyError, ValueError) as e:
            logger.error("Invalid metadata for joint %s: %s for joint %s", joint_name, e, joint_info)
            raise e

        # Determine max_torque based on actuator type
        if "feetech-sts3215" in actuator_type:
            max_torque = sts3215_params["max_torque"]
        elif "feetech-sts3250" in actuator_type:
            max_torque = sts3250_params["max_torque"]
        else:
            logger.error("Unknown actuator type %s for joint %s", actuator_type, joint_name)
            raise ValueError(f"Unknown actuator type {actuator_type} for joint {joint_name}")

        # Configure the actuator through KOS API
        logger.info(
            "Configuring actuator %d (%s): type=%s, kp=%f, kd=%f, max_torque=%f",
            actuator_id,
            joint_name,
            actuator_type,
            kp,
            kd,
            max_torque,
        )
        await kos.actuator.configure_actuator(
            actuator_id=actuator_id,
            kp=kp,
            kd=kd,
            torque_enabled=True,
            max_torque=max_torque,
        )


async def reset(kos: pykos.KOS, actuator_mapping: dict) -> None:
    """Reset the robot to a starting position."""
    await kos.sim.reset(
        pos={"x": 0.0, "y": 0.0, "z": 0.41},
        quat={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        joints=[{"name": info["joint_name"], "pos": 0.0} for info in actuator_mapping.values()],
    )


def spawn_kos_sim(no_render: bool) -> tuple[subprocess.Popen, Callable]:
    """Spawn the KOS-Sim ZBot process and return the process object."""
    logger.info("Starting KOS-Sim zbot-6dof-feet...")
    args = ["kos-sim", "zbot-6dof-feet"]
    if no_render:
        args.append("--no-render")
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    logger.info("Waiting for KOS-Sim to start...")
    time.sleep(5)

    def cleanup(sig: int | None = None, frame: types.FrameType | None = None) -> None:
        logger.info("Terminating KOS-Sim...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        if sig:
            sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)

    return process, cleanup


async def main(
    model_path: str,
    ip: str,
    no_render: bool,
    episode_length: int,
    robot_urdf_path: str,
    actuator_params_path: str,
    metadata_path: str | None = None,
) -> None:
    model = tf.saved_model.load(model_path)
    sim_process = None
    cleanup_fn = None

    try:
        # Try to connect to existing KOS-Sim
        logger.info("Attempting to connect to existing KOS-Sim...")
        kos = pykos.KOS(ip=ip)
        await kos.sim.get_parameters()
        logger.info("Connected to existing KOS-Sim instance.")
    except Exception as e:
        logger.info("Could not connect to existing KOS-Sim: %s", e)
        logger.info("Starting a new KOS-Sim instance locally...")
        sim_process, cleanup_fn = spawn_kos_sim(no_render)
        kos = pykos.KOS()
        attempts = 0
        while attempts < 5:
            try:
                await kos.sim.get_parameters()
                logger.info("Connected to new KOS-Sim instance.")
                break
            except Exception as connect_error:
                attempts += 1
                logger.info("Failed to connect to KOS-Sim: %s", connect_error)
                time.sleep(2)

        if attempts == 5:
            raise RuntimeError("Failed to connect to KOS-Sim")

    # Determine metadata path
    if metadata_path:
        metadata_file = Path(metadata_path)
    else:
        metadata_file = Path(robot_urdf_path) / "metadata.json"

    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_file}")

    actuator_mapping = load_actuator_mapping(metadata_file)
    # Configure actuators with metadata and parameter files
    await configure_actuators(kos, robot_urdf_path, actuator_params_path, metadata_path)
    await reset(kos, actuator_mapping)

    prev_action = np.zeros(len(actuator_mapping))

    observation = (await get_observation(kos, actuator_mapping, prev_action)).reshape(1, -1)

    if no_render:
        await kos.process_manager.start_kclip("deployment")

    # warm up model
    model.infer(observation)

    target_time = time.time() + DT
    observation = await get_observation(kos, actuator_mapping, prev_action)

    end_time = time.time() + episode_length

    try:
        while time.time() < end_time:
            observation = observation.reshape(1, -1)
            # Model only outputs position commands
            action = np.array(model.infer(observation)).reshape(-1)
            observation, _ = await asyncio.gather(
                get_observation(kos, actuator_mapping, prev_action),
                send_actions(kos, action, actuator_mapping),
            )

            prev_action = action
            if time.time() < target_time:
                await asyncio.sleep(max(0, target_time - time.time()))
            else:
                logger.info("Loop overran by %s seconds", time.time() - target_time)

            target_time += DT

    except asyncio.CancelledError:
        logger.info("Exiting...")
        if no_render:
            save_path = await kos.process_manager.stop_kclip("deployment")
            logger.info("KClip saved to %s", save_path)

        if cleanup_fn:
            cleanup_fn()

        raise KeyboardInterrupt

    logger.info("Episode finished!")

    if no_render:
        await kos.process_manager.stop_kclip("deployment")

    if cleanup_fn:
        cleanup_fn()


# (optionally) start the KOS-Sim server before running this script
# `kos-sim zbot2-feet`
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--episode_length", type=int, default=5)  # seconds
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--log-file", type=str, help="Path to write log output")

    # Add arguments to mirror standing.py configuration
    parser.add_argument(
        "--robot_urdf_path",
        type=str,
        default="ksim_zbot/kscale-assets/zbot-6dof-feet/",
        help="The path to the assets directory for the robot.",
    )
    parser.add_argument(
        "--actuator_params_path",
        type=str,
        default="ksim_zbot/kscale-assets/actuators/feetech/",
        help="The path to the assets directory for feetech actuator models",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        help="Path to metadata.json file. If not specified, will look in robot_urdf_path/metadata.json",
    )
    args = parser.parse_args()

    # log_level: logging._Level = logging.DEBUG if args.debug else logging.INFO
    log_level: int = logging.INFO
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if args.log_file:
        logging.basicConfig(
            level=log_level,
            format=log_format,
            filename=args.log_file,
            filemode="w",
        )
    else:
        logging.basicConfig(level=log_level, format=log_format)

    logger.info("Starting deployment with model: %s", args.model_path)
    logger.info("Episode length: %s", args.episode_length)
    logger.info("No render: %s", args.no_render)
    logger.info("Robot URDF path: %s", args.robot_urdf_path)
    logger.info("Actuator params path: %s", args.actuator_params_path)
    logger.info("Metadata path: %s", args.metadata_path)
    logger.info("IP: %s", args.ip)
    logger.info("Debug: %s", args.debug)

    asyncio.run(
        main(
            args.model_path,
            args.ip,
            args.no_render,
            args.episode_length,
            args.robot_urdf_path,
            args.actuator_params_path,
            args.metadata_path,
        )
    )
