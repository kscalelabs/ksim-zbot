"""Example script to deploy a SavedModel on K-Bot."""

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pykos
import tensorflow as tf

logger = logging.getLogger(__name__)

DT = 0.02  # Policy time step (50Hz)
GRAVITY = 9.81  # m/s
ACTION_SCALE = 1.0


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
    Actuator(actuator_id=21, nn_id=0, kp=40.0, kd=4.0, max_torque=40.0, joint_name="dof_right_shoulder_pitch_03"),
    Actuator(actuator_id=22, nn_id=1, kp=40.0, kd=4.0, max_torque=40.0, joint_name="dof_right_shoulder_roll_03"),
    Actuator(actuator_id=23, nn_id=2, kp=30.0, kd=1.0, max_torque=20.0, joint_name="dof_right_shoulder_yaw_02"),
    Actuator(actuator_id=24, nn_id=3, kp=30.0, kd=1.0, max_torque=20.0, joint_name="dof_right_elbow_02"),
    Actuator(
        actuator_id=25, nn_id=4, kp=20.0, kd=0.45473329537059787, max_torque=20.0, joint_name="dof_right_wrist_00"
    ),
    # Left arm (nn_id 5-9)
    Actuator(actuator_id=11, nn_id=5, kp=40.0, kd=4.0, max_torque=40.0, joint_name="dof_left_shoulder_pitch_03"),
    Actuator(actuator_id=12, nn_id=6, kp=40.0, kd=4.0, max_torque=40.0, joint_name="dof_left_shoulder_roll_03"),
    Actuator(actuator_id=13, nn_id=7, kp=30.0, kd=1.0, max_torque=20.0, joint_name="dof_left_shoulder_yaw_02"),
    Actuator(actuator_id=14, nn_id=8, kp=30.0, kd=1.0, max_torque=20.0, joint_name="dof_left_elbow_02"),
    Actuator(actuator_id=15, nn_id=9, kp=20.0, kd=0.45473329537059787, max_torque=20.0, joint_name="dof_left_wrist_00"),
    # Right leg (nn_id 10-14)
    Actuator(actuator_id=41, nn_id=10, kp=85.0, kd=5.0, max_torque=80.0, joint_name="dof_right_hip_pitch_04"),
    Actuator(actuator_id=42, nn_id=11, kp=40.0, kd=4.0, max_torque=40.0, joint_name="dof_right_hip_roll_03"),
    Actuator(actuator_id=43, nn_id=12, kp=40.0, kd=4.0, max_torque=40.0, joint_name="dof_right_hip_yaw_03"),
    Actuator(actuator_id=44, nn_id=13, kp=85.0, kd=5.0, max_torque=80.0, joint_name="dof_right_knee_04"),
    Actuator(actuator_id=45, nn_id=14, kp=30.0, kd=1.0, max_torque=15.0, joint_name="dof_right_ankle_02"),
    # Left leg (nn_id 15-19)
    Actuator(actuator_id=31, nn_id=15, kp=85.0, kd=5.0, max_torque=80.0, joint_name="dof_left_hip_pitch_04"),
    Actuator(actuator_id=32, nn_id=16, kp=40.0, kd=4.0, max_torque=40.0, joint_name="dof_left_hip_roll_03"),
    Actuator(actuator_id=33, nn_id=17, kp=40.0, kd=4.0, max_torque=40.0, joint_name="dof_left_hip_yaw_03"),
    Actuator(actuator_id=34, nn_id=18, kp=85.0, kd=5.0, max_torque=80.0, joint_name="dof_left_knee_04"),
    Actuator(actuator_id=35, nn_id=19, kp=30.0, kd=1.0, max_torque=15.0, joint_name="dof_left_ankle_02"),
]


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

    accel = np.array([imu.accel_y, imu.accel_x, -imu.accel_z]) * GRAVITY

    gyro = np.deg2rad(np.array([imu.gyro_y, imu.gyro_x, -imu.gyro_z]))

    imu_obs = np.concatenate([accel, gyro], axis=-1)
    logger.debug(imu_obs)
    cmd = np.array([0.0, 0.0])
    observation = np.concatenate([pos_obs, vel_obs, imu_obs, cmd], axis=-1)
    return observation


async def send_actions(kos: pykos.KOS, position: np.ndarray, velocity: np.ndarray) -> None:
    position = np.rad2deg(position)
    velocity = np.rad2deg(velocity)
    actuator_commands: list[pykos.services.actuator.ActuatorCommand] = [
        {
            "actuator_id": ac.actuator_id,
            "position": position[ac.nn_id] * ACTION_SCALE,
            "velocity": velocity[ac.nn_id] * ACTION_SCALE,
        }
        for ac in ACTUATOR_LIST
    ]
    # logger.debug(actuator_commands)

    await kos.actuator.command_actuators(actuator_commands)


async def configure_actuators(kos: pykos.KOS, robot_urdf_path: str, actuator_params_path: str) -> None:
    """Configure actuators using parameters from files."""
    # Load the Feetech parameters
    sts3215_params, sts3250_params = load_feetech_params(actuator_params_path)
    
    # Configure each actuator
    for ac in ACTUATOR_LIST:
        joint_name = ac.joint_name
        
        # Determine parameter values based on joint type
        if "_03" in joint_name:  # Assuming _03 corresponds to STS3215
            max_torque = sts3215_params["max_torque"]
        elif "_04" in joint_name:  # Assuming _04 corresponds to STS3250
            max_torque = sts3250_params["max_torque"]
        elif "_02" in joint_name:  # Smaller actuators
            max_torque = ac.max_torque
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
    zero_commands: list[pykos.services.actuator.ActuatorCommand] = [
        {
            "actuator_id": ac.actuator_id,
            "position": 0.0,
            "velocity": 0.0,
        }
        for ac in ACTUATOR_LIST
    ]

    await kos.actuator.command_actuators(zero_commands)


async def disable(kos: pykos.KOS) -> None:
    for ac in ACTUATOR_LIST:
        await kos.actuator.configure_actuator(
            actuator_id=ac.actuator_id,
            torque_enabled=False,
        )


async def main(model_path: str, ip: str, episode_length: int, robot_urdf_path: str, actuator_params_path: str) -> None:
    model = tf.saved_model.load(model_path)
    kos = pykos.KOS(ip=ip)
    await disable(kos)
    time.sleep(1)
    logger.info("Configuring actuators...")
    await configure_actuators(kos, robot_urdf_path, actuator_params_path)
    await asyncio.sleep(1)
    logger.info("Resetting...")
    await reset(kos)

    observation = (await get_observation(kos)).reshape(1, -1)

    # warm up model
    model.infer(observation)

    for i in range(5, -1, -1):
        logger.info("Starting in %d seconds...", i)
        await asyncio.sleep(1)

    target_time = time.time() + DT
    observation = await get_observation(kos)

    end_time = time.time() + episode_length

    try:
        while time.time() < end_time:
            observation = observation.reshape(1, -1)
            # move it all to the infer call
            action = np.array(model.infer(observation)).reshape(-1)
            position = action[: len(ACTUATOR_LIST)]
            velocity = action[len(ACTUATOR_LIST) :]
            observation, _ = await asyncio.gather(
                get_observation(kos),
                send_actions(kos, position, velocity),
            )

            if time.time() < target_time:
                await asyncio.sleep(max(0, target_time - time.time()))
            else:
                logger.info("Loop overran by %s seconds", time.time() - target_time)

            target_time += DT

    except asyncio.CancelledError:
        logger.info("Exiting...")
        await disable(kos)
        logger.info("Actuators disabled")

        raise KeyboardInterrupt

    logger.info("Episode finished!")


# python -m ksim_kbot.kbot2.deploy_real --model_path /path/to/model
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--episode_length", type=int, default=60)  # seconds
    
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

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    asyncio.run(main(
        args.model_path, 
        args.ip, 
        args.episode_length,
        args.robot_urdf_path,
        args.actuator_params_path
    ))
