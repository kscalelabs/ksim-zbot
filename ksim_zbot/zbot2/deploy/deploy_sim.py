"""Example script to deploy a SavedModel in KOS-Sim."""

import argparse
import asyncio
import logging
import signal
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from typing import Callable
from pathlib import Path
import json

import numpy as np
import pykos
import tensorflow as tf
import jax.numpy as jnp

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


ACTUATOR_LIST: list[Actuator] = [
    # Right arm (nn_id 0-4)
    Actuator(actuator_id=21, nn_id=0, kp=200.0, kd=5.0, max_torque=5.47, joint_name="right_shoulder_pitch_15"),
    Actuator(actuator_id=22, nn_id=1, kp=200.0, kd=5.0, max_torque=5.47, joint_name="right_shoulder_roll_15"),
    Actuator(actuator_id=23, nn_id=2, kp=200.0, kd=5.0, max_torque=5.47, joint_name="right_elbow_roll_15"),
    Actuator(actuator_id=24, nn_id=3, kp=200.0, kd=5.0, max_torque=5.47, joint_name="right_gripper_15"),
    Actuator(actuator_id=46, nn_id=4, kp=200.0, kd=5.0, max_torque=8.72, joint_name="right_ankle_roll_50"),
    # Left arm (nn_id 5-9)
    Actuator(actuator_id=11, nn_id=5, kp=200.0, kd=5.0, max_torque=5.47, joint_name="left_shoulder_pitch_15"),
    Actuator(actuator_id=12, nn_id=6, kp=200.0, kd=5.0, max_torque=5.47, joint_name="left_shoulder_roll_15"),
    Actuator(actuator_id=13, nn_id=7, kp=200.0, kd=5.0, max_torque=5.47, joint_name="left_elbow_roll_15"),
    Actuator(actuator_id=14, nn_id=8, kp=200.0, kd=5.0, max_torque=5.47, joint_name="left_gripper_15"),
    Actuator(actuator_id=36, nn_id=9, kp=200.0, kd=5.0, max_torque=8.72, joint_name="left_ankle_roll_50"),
    # Right leg (nn_id 10-14)
    Actuator(actuator_id=43, nn_id=10, kp=200.0, kd=5.0, max_torque=8.72, joint_name="right_hip_pitch_50"),
    Actuator(actuator_id=42, nn_id=11, kp=200.0, kd=5.0, max_torque=8.72, joint_name="right_hip_roll_50"),
    Actuator(actuator_id=41, nn_id=12, kp=200.0, kd=5.0, max_torque=8.72, joint_name="right_hip_yaw_50"),
    Actuator(actuator_id=44, nn_id=13, kp=200.0, kd=5.0, max_torque=8.72, joint_name="right_knee_pitch_50"),
    Actuator(actuator_id=45, nn_id=14, kp=200.0, kd=5.0, max_torque=8.72, joint_name="right_ankle_pitch_50"),
    # Left leg (nn_id 15-19)
    Actuator(actuator_id=33, nn_id=15, kp=200.0, kd=5.0, max_torque=8.72, joint_name="left_hip_pitch_50"),
    Actuator(actuator_id=32, nn_id=16, kp=200.0, kd=5.0, max_torque=8.72, joint_name="left_hip_roll_50"),
    Actuator(actuator_id=31, nn_id=17, kp=200.0, kd=5.0, max_torque=8.72, joint_name="left_hip_yaw_50"),
    Actuator(actuator_id=34, nn_id=18, kp=200.0, kd=5.0, max_torque=8.72, joint_name="left_knee_pitch_50"),
    Actuator(actuator_id=35, nn_id=19, kp=200.0, kd=5.0, max_torque=8.72, joint_name="left_ankle_pitch_50"),
]


async def get_observation(kos: pykos.KOS) -> np.ndarray:
    (actuator_states, imu) = await asyncio.gather(
        kos.actuator.get_actuators_state([ac.actuator_id for ac in ACTUATOR_LIST]),
        kos.imu.get_imu_values(),
    )
    state_dict_pos = {state.actuator_id: state.position for state in actuator_states.states}
    pos_obs = np.deg2rad(
        np.array([state_dict_pos[ac.actuator_id] for ac in sorted(ACTUATOR_LIST, key=lambda x: x.nn_id)])
    )
    state_dict_vel = {state.actuator_id: state.velocity for state in actuator_states.states}
    vel_obs = np.deg2rad(
        np.array([state_dict_vel[ac.actuator_id] for ac in sorted(ACTUATOR_LIST, key=lambda x: x.nn_id)])
    )
    imu_obs = np.array([imu.accel_x, imu.accel_y, imu.accel_z, imu.gyro_x, imu.gyro_y, imu.gyro_z])
    cmd = np.array([0.0, 0.0])
    last_action = np.zeros(len(ACTUATOR_LIST))  # Add last action to observation
    observation = np.concatenate([pos_obs, vel_obs, imu_obs, cmd, last_action], axis=-1)
    return observation


async def send_actions(kos: pykos.KOS, position: np.ndarray) -> None:
    position = np.rad2deg(position)
    actuator_commands: list[pykos.services.actuator.ActuatorCommand] = [
        {
            "actuator_id": ac.actuator_id,
            "position": position[ac.nn_id],
            "velocity": 0.0,  # Set velocity to 0 since we're only using position control
        }
        for ac in ACTUATOR_LIST
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


async def configure_actuators(kos: pykos.KOS, robot_urdf_path: str, actuator_params_path: str) -> None:
    """Configure actuators using parameters from files."""
    # Load the Feetech parameters
    sts3215_params, sts3250_params = load_feetech_params(actuator_params_path)
    
    # Configure each actuator
    for ac in ACTUATOR_LIST:
        joint_name = ac.joint_name
        
        # Determine parameter values based on joint type
        if "_15" in joint_name:
            max_torque = sts3215_params["max_torque"]
        elif "_50" in joint_name:
            max_torque = sts3250_params["max_torque"]
        else:
            max_torque = ac.max_torque
        
        # Use the predefined kp/kd values from ACTUATOR_LIST
        kp = ac.kp
        kd = ac.kd
        
        # Configure the actuator through KOS API
        logger.info(f"Configuring actuator {ac.actuator_id} ({joint_name}): kp={kp}, kd={kd}, max_torque={max_torque}")
        await kos.actuator.configure_actuator(
            actuator_id=ac.actuator_id,
            kp=kp,
            kd=kd,
            torque_enabled=True,
            max_torque=max_torque,
        )


async def reset(kos: pykos.KOS) -> None:
    await kos.sim.reset(
        pos={"x": 0.0, "y": 0.0, "z": 0.41024304},
        quat={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        joints=[
            {"name": actuator.joint_name, "pos": pos}
            for actuator, pos in zip(ACTUATOR_LIST, [0.0] * len(ACTUATOR_LIST))
        ],
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
    actuator_params_path: str
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

    # Configure actuators with metadata and parameter files
    await configure_actuators(kos, robot_urdf_path, actuator_params_path)
    await reset(kos)

    observation = (await get_observation(kos)).reshape(1, -1)

    if no_render:
        await kos.process_manager.start_kclip("deployment")

    # warm up model
    model.infer(observation)

    target_time = time.time() + DT
    observation = await get_observation(kos)
    last_action = np.zeros(len(ACTUATOR_LIST))

    end_time = time.time() + episode_length

    try:
        while time.time() < end_time:
            observation = observation.reshape(1, -1)
            # Model only outputs position commands
            position = np.array(model.infer(observation)).reshape(-1)
            observation, _ = await asyncio.gather(
                get_observation(kos),
                send_actions(kos, position),
            )
            last_action = position  # Store last action for next observation

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
# `kos-sim kbot2-feet`
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
        help="The path to the assets directory for the robot."
    )
    parser.add_argument(
        "--actuator_params_path", 
        type=str, 
        default="ksim_zbot/kscale-assets/actuators/feetech/",
        help="The path to the assets directory for feetech actuator models"
    )
    args = parser.parse_args()

    # log_level: logging._Level = logging.DEBUG if args.debug else logging.INFO
    log_level: int = logging.DEBUG
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

    asyncio.run(main(
        args.model_path, 
        args.ip, 
        args.no_render, 
        args.episode_length,
        args.robot_urdf_path,
        args.actuator_params_path
    ))
