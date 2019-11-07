# This file is part of ts_rotator.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License

__all__ = ["MockMTHexapodController"]

import math

from lsst.ts import hexrotcomm
from lsst.ts.idl.enums import Hexapod
from . import constants
from . import enums
from . import structs
from . import utils
from . import simple_hexapod

# Maximum time between track commands (seconds)
# The real controller may use 0.15
TRACK_TIMEOUT = 1


class MockMTHexapodController(hexrotcomm.BaseMockController):
    """Mock MT rotator controller that talks over TCP/IP.

    Parameters
    ----------
    index : `SalIndex` or `int`
        SAL index; see `SalIndex` for the allowed values.
    log : `logging.Logger`
        Logger.
    command_port : `int` (optional)
        Command socket port.  This argument is intended for unit tests;
        use the default value for normal operation.
    telemetry_port : `int` (optional)
        Telemetry socket port. This argument is intended for unit tests;
        use the default value for normal operation.
    initial_state : `Hexapod.ControllerState` (optional)
        Initial state of mock controller.

    Notes
    -----
    To start the mock controller:

        ctrl = MockHexapodController(...)
        await ctrl.connect_task

    To stop the server:

        await ctrl.stop()

    *Known Limitations*

    * The synchronized move parameter is ignored.
      Supporting this would be fairly easy.
    * No lookup table support. Thus MOVE_LUT is the same as
      MOVE_POINT_TO_POINT.
    * Acceleration is treated as instantanous.
    """
    # Arguments for SimpleHexapod.make_zigzag_model
    base_radius = 1e6
    mirror_radius = 1e6
    mirror_z = 0.5e6
    base_angle0 = 0
    pivot = (0, 0, mirror_z + 1e5)
    max_length = mirror_z * 3
    # Set speed such that reasonably large moves take about a second
    # (for unit testing).
    speed = mirror_z/500

    def __init__(self, index, log, command_port, telemetry_port,
                 initial_state=Hexapod.ControllerState.OFFLINE):
        self.encoder_resolution = 200_000  # counts/deg; arbitrary
        index = enums.SalIndex(index)
        self.xy_max_limit = constants.XY_MAX_LIMIT[index-1]
        self.z_min_limit = constants.Z_MIN_LIMIT[index-1]
        self.z_max_limit = constants.Z_MAX_LIMIT[index-1]
        self.uv_max_limit = constants.UV_MAX_LIMIT[index-1]
        self.w_min_limit = constants.W_MIN_LIMIT[index-1]
        self.w_max_limit = constants.W_MAX_LIMIT[index-1]
        config = structs.Config()
        config.strut_acceleration = 500
        # Order: xy (um), zmin, max, uv (deg), wmin, wmax
        config.pos_limits = (self.xy_max_limit, self.z_min_limit, self.z_max_limit,
                             self.uv_max_limit, self.w_min_limit, self.w_max_limit)
        # Order: xy (deg/sec), xy rotation (um/sec), z, z rotation
        config.vel_limits = (constants.MAX_LINEAR_VEL_LIMIT, constants.MAX_ANGULAR_VEL_LIMIT,
                             constants.MAX_LINEAR_VEL_LIMIT, constants.MAX_ANGULAR_VEL_LIMIT)
        # Order: x, y, z, u, w, v
        config.initial_pos = (0, 0, 0, 0, 0, 0)
        config.pivot = self.pivot
        config.strut_displacement_max = self.max_length
        config.strut_velocity_max = self.speed

        self.hexapod = simple_hexapod.SimpleHexapod.make_zigzag_model(
            base_radius=self.base_radius,
            mirror_radius=self.mirror_radius,
            mirror_z=self.mirror_z,
            base_angle0=self.base_angle0,
            pivot=self.pivot,
            min_length=0,
            max_length=self.max_length,
            speed=self.speed,
        )

        telemetry = structs.Telemetry()
        telemetry.commanded_pos = (0,)*6
        self.set_position = (math.nan,)*6

        # Dict of command key: command
        self.command_table = {
            (enums.CommandCode.SET_STATE, enums.SetStateParam.START): self.do_start,
            (enums.CommandCode.SET_STATE, enums.SetStateParam.ENABLE): self.do_enable,
            (enums.CommandCode.SET_STATE, enums.SetStateParam.STANDBY): self.do_standby,
            (enums.CommandCode.SET_STATE, enums.SetStateParam.DISABLE): self.do_disable,
            (enums.CommandCode.SET_STATE, enums.SetStateParam.EXIT): self.do_exit,
            (enums.CommandCode.SET_STATE, enums.SetStateParam.CLEAR_ERROR): self.do_clear_error,
            (enums.CommandCode.SET_STATE, enums.SetStateParam.ENTER_CONTROL): self.do_enter_control,
            (enums.CommandCode.SET_ENABLED_SUBSTATE,
             enums.SetEnabledSubstateParam.MOVE_POINT_TO_POINT): self.do_move_point_to_point,
            # Note: the mock controller ignores the lookup table,
            # so MOVE_LUT is identical to MOVE_POINT_TO_POINT
            (enums.CommandCode.SET_ENABLED_SUBSTATE,
             enums.SetEnabledSubstateParam.MOVE_LUT): self.do_move_point_to_point,
            (enums.CommandCode.SET_ENABLED_SUBSTATE, enums.SetEnabledSubstateParam.STOP): self.do_stop,
            enums.CommandCode.POSITION_SET: self.do_position_set,
            enums.CommandCode.SET_PIVOTPOINT: self.do_set_pivotpoint,
            enums.CommandCode.CONFIG_ACCEL: self.do_config_accel,
            enums.CommandCode.CONFIG_LIMITS: self.do_config_limits,
            enums.CommandCode.CONFIG_VEL: self.do_config_vel,
        }

        super().__init__(log=log, config=config, telemetry=telemetry,
                         command_port=command_port, telemetry_port=telemetry_port)
        self.set_state(initial_state)

    @property
    def state(self):
        return self.telemetry.state

    @property
    def offline_substate(self):
        return self.telemetry.offline_substate

    @property
    def enabled_substate(self):
        return self.telemetry.enabled_substate

    async def close(self):
        """Kill command and telemetry tasks and close the connections.

        Always safe to call.
        """
        self.hexapod.stop()
        await super().close()

    def assert_stationary(self):
        self.assert_state(Hexapod.ControllerState.ENABLED,
                          enabled_substate=Hexapod.EnabledSubstate.STATIONARY)

    def assert_state(self, state, offline_substate=None, enabled_substate=None):
        if self.state != state:
            raise RuntimeError(f"state={self.state!r}; must be {state!r} for this command.")
        if offline_substate is not None and self.offline_substate != offline_substate:
            raise RuntimeError(f"offline_substate={self.offline_substate!r}; "
                               f"must be {offline_substate!r} for this command.")
        if enabled_substate is not None and self.enabled_substate != enabled_substate:
            raise RuntimeError(f"enabled_substate={self.enabled_substate!r}; "
                               f"must be {enabled_substate!r} for this command.")

    def get_command_key(self, command):
        """Return the key to command_table."""
        if command.cmd in (enums.CommandCode.SET_STATE,
                           enums.CommandCode.SET_ENABLED_SUBSTATE):
            return (command.cmd, int(command.param1))
        return command.cmd

    async def do_enter_control(self, command):
        self.assert_state(Hexapod.ControllerState.OFFLINE,
                          offline_substate=Hexapod.OfflineSubstate.AVAILABLE)
        self.set_state(Hexapod.ControllerState.STANDBY)

    async def do_start(self, command):
        self.assert_state(Hexapod.ControllerState.STANDBY)
        self.set_state(Hexapod.ControllerState.DISABLED)

    async def do_enable(self, command):
        self.assert_state(Hexapod.ControllerState.DISABLED)
        self.set_state(Hexapod.ControllerState.ENABLED)

    async def do_disable(self, command):
        self.assert_state(Hexapod.ControllerState.ENABLED)
        self.set_state(Hexapod.ControllerState.DISABLED)

    async def do_standby(self, command):
        self.assert_state(Hexapod.ControllerState.DISABLED)
        self.set_state(Hexapod.ControllerState.STANDBY)

    async def do_exit(self, command):
        self.assert_state(Hexapod.ControllerState.STANDBY)
        self.set_state(Hexapod.ControllerState.OFFLINE)

    async def do_clear_error(self, command):
        # Allow initial state FAULT and OFFLINE because the real controller
        # requires two sequential CLEAR_COMMAND commands. For the mock
        # controller the first command will (probably) transition from FAULT
        # to OFFLINE, but the second must be accepted without complaint.
        if self.state not in (Hexapod.ControllerState.FAULT, Hexapod.ControllerState.OFFLINE):
            raise RuntimeError(f"state={self.state!r}; must be FAULT or OFFLINE for this command.")
        self.set_state(Hexapod.ControllerState.OFFLINE)

    async def do_config_accel(self, command):
        self.assert_stationary()
        if not 0 < command.param1 <= constants.MAX_ACCEL_LIMIT:
            raise ValueError(f"Requested accel limit {command.param1} "
                             f"not in range (0, {constants.MAX_ACCEL_LIMIT}]")
        self.config.strut_acceleration = command.param1
        await self.write_config()

    async def do_config_limits(self, command):
        self.assert_stationary()
        utils.check_positive_value(command.param1, "xymax", self.xy_max_limit)
        utils.check_negative_value(command.param2, "zmin", self.z_min_limit)
        utils.check_positive_value(command.param3, "zmax", self.z_max_limit)
        utils.check_positive_value(command.param4, "uvmax", self.uv_max_limit)
        utils.check_negative_value(command.param5, "wmin", self.w_min_limit)
        utils.check_positive_value(command.param6, "wmax", self.w_max_limit)
        self.config.pos_limits = (command.param1, command.param2, command.param3,
                                  command.param4, command.param5, command.param6)
        await self.write_config()

    async def do_config_vel(self, command):
        self.assert_stationary()
        utils.check_positive_value(command.param1, "xymax", constants.MAX_LINEAR_VEL_LIMIT)
        utils.check_positive_value(command.param2, "rxrymax", constants.MAX_ANGULAR_VEL_LIMIT)
        utils.check_positive_value(command.param3, "zmax", constants.MAX_LINEAR_VEL_LIMIT)
        utils.check_positive_value(command.param4, "rzmax", constants.MAX_ANGULAR_VEL_LIMIT)
        self.config.vel_limits = (command.param1, command.param2,
                                  command.param3, command.param4)
        await self.write_config()

    async def do_offset(self, command):
        self.assert_stationary()

    async def do_position_set(self, command):
        self.assert_stationary()
        utils.check_symmetrical_range(command.param1, "x", self.config.pos_limits[0])
        utils.check_symmetrical_range(command.param2, "y", self.config.pos_limits[0])
        utils.check_range(command.param3, "z", self.config.pos_limits[1],
                          self.config.pos_limits[2])
        utils.check_symmetrical_range(command.param4, "u", self.config.pos_limits[3])
        utils.check_symmetrical_range(command.param5, "v", self.config.pos_limits[3])
        utils.check_range(command.param6, "w", self.config.pos_limits[4],
                          self.config.pos_limits[5])
        self.set_position = (command.param1, command.param2, command.param3,
                             command.param4, command.param5, command.param6)

    async def do_set_pivotpoint(self, command):
        self.assert_stationary()
        self.config.pivot = (command.param1, command.param2, command.param3)
        await self.write_config()

    async def do_stop(self, command):
        self.assert_state(Hexapod.ControllerState.ENABLED)
        self.hexapod.stop()
        self.telemetry.enabled_substate = Hexapod.EnabledSubstate.STATIONARY

    async def do_move_point_to_point(self, command):
        if not math.isfinite(self.set_position[0]):
            raise RuntimeError("Must call POSITION_SET before calling MOVE_POINT_TO_POINT")
        self.telemetry.commanded_pos = self.set_position
        self.hexapod.move(pos=self.telemetry.commanded_pos[0:3],
                          xyzrot=self.telemetry.commanded_pos[3:6])
        self.telemetry.commanded_length = tuple(actuator.end_pos for actuator in self.hexapod.actuators)
        self.telemetry.enabled_substate = Hexapod.EnabledSubstate.MOVING_POINT_TO_POINT

    async def run_command(self, command):
        self.log.debug(f"run_command: command={enums.CommandCode(command.cmd)!r}; "
                       f"param1={command.param1}; param2={command.param2}; param3={command.param3}")
        key = self.get_command_key(command)
        cmd_method = self.command_table.get(key, None)
        if cmd_method is None:
            self.log.error(f"Unrecognized command cmd={command.cmd}; param1={command.param1}")
            return
        try:
            await cmd_method(command)
        except Exception as e:
            self.log.error(f"Command cmd={command.cmd}; param1={command.param1} failed: {e}")
        if cmd_method != self.do_position_set:
            self.set_position = (math.nan,)*6

    def set_state(self, state):
        """Set the current state and substates.

        Parameters
        ----------
        state : `Hexapod.ControllerState` or `int`
            New state.

        Notes
        -----
        Sets the substates as follows:

        * `Hexapod.OfflineSubstate.AVAILABLE`
          if state == `Hexapod.ControllerState.OFFLINE`
        * `Hexapod.EnabledSubstate.STATIONARY`
          if state == `Hexapod.ControllerState.ENABLED`

        The real controller goes to substate
        `lsst.ts.idl.enums.Hexapod.OfflineSubstate.PUBLISH_ONLY` when going
        offline, but requires the engineering user interface (EUI) to get out
        of that state, and we don't have an EUI for the mock controller!
        """
        self.telemetry.state = Hexapod.ControllerState(state)
        self.telemetry.offline_substate = Hexapod.OfflineSubstate.AVAILABLE \
            if self.telemetry.state == Hexapod.ControllerState.OFFLINE else 0
        self.telemetry.enabled_substate = Hexapod.EnabledSubstate.STATIONARY \
            if self.telemetry.state == Hexapod.ControllerState.ENABLED else 0
        self.log.debug(f"set_state: state={Hexapod.ControllerState(self.telemetry.state)!r}; "
                       f"offline_substate={Hexapod.EnabledSubstate(self.telemetry.offline_substate)}; "
                       f"enabled_substate={Hexapod.EnabledSubstate(self.telemetry.enabled_substate)}")

    def log_rejected_command(self, command, reason=None):
        """Log an error message describing a rejected command."""
        if reason is None:
            reason = f"the current state is {self.telemetry.state}, " \
                     f"enabled_substate={self.telemetry.enabled_substate}"
        self.log.error(f"Ignoring command cmd={command.cmd}, "
                       f"param1={command.param1}, param2={command.param2}: {reason}")

    async def update_telemetry(self):
        try:
            self.telemetry.status_word = (0,)*6
            self.telemetry.latching_fault_status_register = (0,)*6
            self.telemetry.copley_fault_status_register = (0,)*6
            axes_in_position = [not actuator.moving for actuator in self.hexapod.actuators]
            self.telemetry.application_status = \
                tuple(int(in_position)*Hexapod.ApplicationStatus.HEX_MOVE_COMPLETE_MASK
                      for in_position in axes_in_position)
            self.telemetry.input_pin_states = (0,)*3

            # state, enabled_substate and offline_substate
            # are all set by set_state
            self.telemetry.test_state = 0

            curr_pos = [actuator.curr_pos for actuator in self.hexapod.actuators]
            self.telemetry.strut_encoder_raw = tuple(pos*self.encoder_resolution for pos in curr_pos)
            self.telemetry.strut_encoder_microns = tuple(curr_pos)
            # self.telemetry.commanded_pos and commanded_length are both set
            # by MOVE and MOVE_LUT.
            # self.telemetry.measured_pos should be based on current position,
            # but SimpleHexapod does not yet support determining orientation
            # from actuator length
            self.telemetry.measured_pos = tuple(self.telemetry.commanded_pos[:])

            if self.telemetry.state == Hexapod.ControllerState.ENABLED and \
                    self.telemetry.enabled_substate == Hexapod.EnabledSubstate.MOVING_POINT_TO_POINT and \
                    all(axes_in_position):
                self.telemetry.enabled_substate = Hexapod.EnabledSubstate.STATIONARY
        except Exception:
            self.log.exception("update_telemetry failed; output incomplete telemetry")
