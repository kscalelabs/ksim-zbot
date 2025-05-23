"""This file contains tests for the FeetechActuators class.

Run with python -m pytest -rP to see the print statements.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from ksim_zbot.zbot2.common import FeetechActuators

# Base path for test assets
ASSETS_DIR = Path(__file__).parent / "assets/actuators/"

# Parameter filenames to test
PARAMS_FILES = ["feetech_sts3250.json", "feetech_sts3215_12v.json"]

# --- Module-level check for parameter files ---
# Construct full paths and fail early if any are missing
for filename in PARAMS_FILES:
    params_path = ASSETS_DIR / filename
    if not params_path.is_file():
        raise FileNotFoundError(f"Required actuator params file not found during test setup: {params_path}")


NUM_ACTUATORS = 1


@pytest.fixture
def real_feetech_params(request: pytest.FixtureRequest) -> dict:
    """Load Feetech parameters from the specified JSON file."""
    filename = request.param
    params_file_path = ASSETS_DIR / filename

    if not params_file_path.exists():
        raise FileNotFoundError(f"Parameter file not found: {params_file_path}")

    with open(params_file_path, "r") as f:
        params = json.load(f)

    required_keys = ["max_torque", "max_velocity", "max_pwm", "vin", "kt", "R", "error_gain"]
    for key in required_keys:
        if key not in params:
            raise ValueError(f"Parameter file {params_file_path} is missing required key: {key}")

    params["_filename"] = filename
    return params


class MockPhysicsData:
    """Mock class for physics data used in testing."""

    def __init__(self) -> None:
        self.qpos = jnp.zeros(NUM_ACTUATORS + 7)
        self.qvel = jnp.zeros(NUM_ACTUATORS + 6)


@pytest.fixture
def mock_physics_data() -> MockPhysicsData:
    """Fixture that provides a MockPhysicsData instance."""
    return MockPhysicsData()


@pytest.mark.parametrize("real_feetech_params", PARAMS_FILES, indirect=True)
@pytest.mark.parametrize("jit_enabled", [True, False])
def test_actuator_basic_functionality(
    real_feetech_params: dict,
    mock_physics_data: MockPhysicsData,
    jit_enabled: bool,
) -> None:
    """Test basic actuator functionality with both JIT enabled and disabled."""
    # Configure JIT based on parameter
    if not jit_enabled:
        jax.config.update("jax_disable_jit", True)
    else:
        jax.config.update("jax_disable_jit", False)

    num_actuators = NUM_ACTUATORS
    assert num_actuators == len(mock_physics_data.qpos) - 7  # Subtract 7 base positions
    assert num_actuators == len(mock_physics_data.qvel) - 6  # Subtract 6 base velocities

    params = real_feetech_params
    print(
        f"--- Testing with parameters from: {params['_filename']} (JIT {'enabled' if jit_enabled else 'disabled'}) ---"
    )

    # kp and kd are not directly in the params file, using placeholder values for now
    # In a real scenario, these would likely come from metadata or config
    kp_j = jnp.array([20.0] * num_actuators)
    kd_j = jnp.array([5.0] * num_actuators)
    vmax_j = jnp.array([5.0] * num_actuators)
    amax_j = jnp.array([39.0] * num_actuators)

    max_torque_j = jnp.array([params["max_torque"]] * num_actuators)
    max_velocity_j = jnp.array([params["max_velocity"]] * num_actuators)
    max_pwm_j = jnp.array([params["max_pwm"]] * num_actuators)
    vin_j = jnp.array([params["vin"]] * num_actuators)
    kt_j = jnp.array([params["kt"]] * num_actuators)
    r_j = jnp.array([params["R"]] * num_actuators)
    error_gain_j = jnp.array([params["error_gain"]] * num_actuators)

    actuators = FeetechActuators(
        max_torque_j=max_torque_j,
        kp_j=kp_j,
        kd_j=kd_j,
        max_velocity_j=max_velocity_j,
        max_pwm_j=max_pwm_j,
        vin_j=vin_j,
        kt_j=kt_j,
        r_j=r_j,
        vmax_j=vmax_j,
        amax_j=amax_j,
        dt=0.001,
        error_gain_j=error_gain_j,
    )

    action_j = jnp.ones(num_actuators)
    print(f"Input action_j: {action_j}")

    rng = jax.random.PRNGKey(0)

    # Get initial state for the planner
    initial_pos_j = mock_physics_data.qpos[7:]
    initial_vel_j = mock_physics_data.qvel[6:]
    initial_planner_state = actuators.get_default_state(initial_pos_j, initial_vel_j)

    # Use get_stateful_ctrl instead of get_ctrl
    torque_j, _ = actuators.get_stateful_ctrl(action_j, mock_physics_data, initial_planner_state, rng)

    print(f"Generated torque_j: {torque_j}")

    assert torque_j.shape == (num_actuators,)
    assert isinstance(torque_j, jnp.ndarray)

    # Reset JIT configuration after test
    jax.config.update("jax_disable_jit", False)
