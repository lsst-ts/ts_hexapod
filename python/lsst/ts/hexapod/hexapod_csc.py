# This file is part of ts_hexapod.
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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import copy
import os
import pathlib

from lsst.ts import salobj
from lsst.ts import hexrotcomm
from lsst.ts.idl.enums import Hexapod
from . import constants
from . import enums
from . import structs
from . import utils
from . import mock_controller
from .lookup_table import LookupTable


class ControllerConstants:
    """Constants needed to communicate with a hexapod controller.
    """

    def __init__(self, sync_pattern, config_frame_id, telemetry_frame_id):
        self.sync_pattern = sync_pattern
        self.config_frame_id = config_frame_id
        self.telemetry_frame_id = telemetry_frame_id


# Dict of SalIndex: ControllerConstants
IndexControllerConstants = {
    enums.SalIndex.CAM_HEXAPOD: ControllerConstants(
        sync_pattern=constants.CAM_SYNC_PATTERN,
        config_frame_id=enums.FrameId.CAM_CONFIG,
        telemetry_frame_id=enums.FrameId.CAM_TELEMETRY,
    ),
    enums.SalIndex.M2_HEXAPOD: ControllerConstants(
        sync_pattern=constants.M2_SYNC_PATTERN,
        config_frame_id=enums.FrameId.M2_CONFIG,
        telemetry_frame_id=enums.FrameId.M2_TELEMETRY,
    ),
}


class HexapodCsc(hexrotcomm.BaseCsc):
    """MT hexapod CSC.

    Parameters
    ----------
    index : `SalIndex` or `int`
        SAL index; see `SalIndex` for the allowed values.
    initial_state : `lsst.ts.salobj.State` or `int` (optional)
        The initial state of the CSC. Ignored (other than checking
        that it is a valid value) except in simulation mode,
        because in normal operation the initial state is the current state
        of the controller. This is provided for unit testing.
    simulation_mode : `int` (optional)
        Simulation mode. Allowed values:

        * 0: regular operation.
        * 1: simulation: use a mock low level controller.
    compensation_data_dir : `str`, optional
        Directory of compensation data files, or None for
        the standard directory (obtained from `get_compensation_data_dir`).
        This is provided for unit testing.

    Notes
    -----
    **Error Codes**

    * 1: invalid data read on the telemetry socket

    This CSC is unusual in several respect:

    * It acts as a server (not a client) for a low level controller
      (because that is how the low level controller is written).
    * The low level controller maintains the summary state and detailed state
      (that's why this code inherits from Controller instead of BaseCsc).
    * The simulation mode can only be set at construction time.
    """

    def __init__(
        self,
        index,
        initial_state=salobj.State.OFFLINE,
        compensation_data_dir=None,
        simulation_mode=0,
    ):
        index = enums.SalIndex(index)
        controller_constants = IndexControllerConstants[index]
        self.xy_max_limit = constants.XY_MAX_LIMIT[index - 1]
        self.z_min_limit = constants.Z_MIN_LIMIT[index - 1]
        self.z_max_limit = constants.Z_MAX_LIMIT[index - 1]
        self.uv_max_limit = constants.UV_MAX_LIMIT[index - 1]
        self.w_min_limit = constants.W_MIN_LIMIT[index - 1]
        self.w_max_limit = constants.W_MAX_LIMIT[index - 1]

        structs.Config.FRAME_ID = controller_constants.config_frame_id
        structs.Telemetry.FRAME_ID = controller_constants.telemetry_frame_id

        if compensation_data_dir is None:
            self.compensation_data_dir = self.get_compensation_data_dir(index=index)
        else:
            if not pathlib.Path(compensation_data_dir).isdir():
                raise ValueError(
                    f"compensation_data_dir={compensation_data_dir} does not exist"
                )
            self.compensation_data_dir = compensation_data_dir

        self.load_luts(
            azimuth_prefix="default",
            elevation_prefix="default",
            temperature_prefix="default",
        )

        super().__init__(
            name="Hexapod",
            index=index,
            sync_pattern=controller_constants.sync_pattern,
            CommandCode=enums.CommandCode,
            ConfigClass=structs.Config,
            TelemetryClass=structs.Telemetry,
            initial_state=initial_state,
            simulation_mode=simulation_mode,
        )

    # Hexapod-specific commands.
    async def do_compensatedMove(self, data):
        """Move to a specified position and orientation,
        with compensation for telescope elevation, azimuth and temperature.
        """
        self.assert_enabled_substate(Hexapod.EnabledSubstate.STATIONARY)
        cmd1 = self._make_position_set_command(data)
        cmd2 = self.make_command(
            code=enums.CommandCode.SET_ENABLED_SUBSTATE,
            param1=enums.SetEnabledSubstateParam.MOVE_LUT,
            param2=data.sync,
            param3=data.azimuth,
            param4=data.elevation,
            param5=data.temperature,
        )
        await self.run_multiple_commands(cmd1, cmd2)

    async def do_configureAcceleration(self, data):
        """Specify the acceleration limit."""
        self.assert_enabled_substate(Hexapod.EnabledSubstate.STATIONARY)
        utils.check_positive_value(
            data.acceleration,
            "acceleration",
            constants.MAX_ACCEL_LIMIT,
            ExceptionClass=salobj.ExpectedError,
        )
        await self.run_command(
            code=enums.CommandCode.CONFIG_ACCEL, param1=data.acceleration
        )

    async def do_configureLimits(self, data):
        """Specify position and rotation limits."""
        self.assert_enabled_substate(Hexapod.EnabledSubstate.STATIONARY)
        utils.check_positive_value(
            data.maxXY, "maxXY", self.xy_max_limit, ExceptionClass=salobj.ExpectedError
        )
        utils.check_negative_value(
            data.minZ, "minZ", self.z_min_limit, ExceptionClass=salobj.ExpectedError
        )
        utils.check_positive_value(
            data.maxZ, "maxZ", self.z_max_limit, ExceptionClass=salobj.ExpectedError
        )
        utils.check_positive_value(
            data.maxUV, "maxUV", self.uv_max_limit, ExceptionClass=salobj.ExpectedError
        )
        utils.check_negative_value(
            data.minW, "minW", self.w_min_limit, ExceptionClass=salobj.ExpectedError
        )
        utils.check_positive_value(
            data.maxW, "maxW", self.w_max_limit, ExceptionClass=salobj.ExpectedError
        )
        await self.run_command(
            code=enums.CommandCode.CONFIG_LIMITS,
            param1=data.maxXY,
            param2=data.minZ,
            param3=data.maxZ,
            param4=data.maxUV,
            param5=data.minW,
            param6=data.maxW,
        )

    async def do_configureVelocity(self, data):
        """Specify velocity limits."""
        self.assert_enabled_substate(Hexapod.EnabledSubstate.STATIONARY)
        utils.check_positive_value(
            data.xy,
            "xy",
            constants.MAX_LINEAR_VEL_LIMIT,
            ExceptionClass=salobj.ExpectedError,
        )
        utils.check_positive_value(
            data.z,
            "z",
            constants.MAX_LINEAR_VEL_LIMIT,
            ExceptionClass=salobj.ExpectedError,
        )
        utils.check_positive_value(
            data.uv,
            "uv",
            constants.MAX_ANGULAR_VEL_LIMIT,
            ExceptionClass=salobj.ExpectedError,
        )
        utils.check_positive_value(
            data.w,
            "w",
            constants.MAX_ANGULAR_VEL_LIMIT,
            ExceptionClass=salobj.ExpectedError,
        )
        await self.run_command(
            code=enums.CommandCode.CONFIG_VEL,
            param1=data.xy,
            param2=data.uv,
            param3=data.z,
            param4=data.w,
        )

    async def do_loadCompensationFiles(self, data):
        self.assert_enabled_substate(Hexapod.EnabledSubstate.STATIONARY)
        self.load_luts(
            elevation_prefix=data.elevationPrefix,
            azimuth_prefix=data.azimuthPrefix,
            temperature_prefix=data.temperaturePrefix,
        )

    async def do_move(self, data):
        """Move to a specified position and orientation.
        """
        self.assert_enabled_substate(Hexapod.EnabledSubstate.STATIONARY)
        cmd1 = self._make_position_set_command(data)
        cmd2 = self.make_command(
            code=enums.CommandCode.SET_ENABLED_SUBSTATE,
            param1=enums.SetEnabledSubstateParam.MOVE_POINT_TO_POINT,
            param2=data.sync,
        )
        await self.run_multiple_commands(cmd1, cmd2)

    async def do_offset(self, data):
        """Move by a specified offset in position and orientation.
        """
        self.assert_enabled_substate(Hexapod.EnabledSubstate.STATIONARY)
        cmd1 = self._make_offset_set_command(data)
        cmd2 = self.make_command(
            code=enums.CommandCode.SET_ENABLED_SUBSTATE,
            param1=enums.SetEnabledSubstateParam.MOVE_POINT_TO_POINT,
            param2=data.sync,
        )
        await self.run_multiple_commands(cmd1, cmd2)

    async def do_pivot(self, data):
        """Set the coordinates of the pivot point.
        """
        self.assert_enabled_substate(Hexapod.EnabledSubstate.STATIONARY)
        await self.run_command(
            code=enums.CommandCode.SET_PIVOTPOINT,
            param1=data.x,
            param2=data.y,
            param3=data.z,
        )

    async def do_stop(self, data):
        """Halt tracking or any other motion.
        """
        if self.summary_state != salobj.State.ENABLED:
            raise salobj.ExpectedError("Not enabled")
        await self.run_command(
            code=enums.CommandCode.SET_ENABLED_SUBSTATE,
            param1=enums.SetEnabledSubstateParam.STOP,
        )

    def config_callback(self, server):
        """Called when the TCP/IP controller outputs configuration.

        Parameters
        ----------
        server : `lsst.ts.hexrotcomm.CommandTelemetryServer`
            TCP/IP server.
        """
        self.evt_configuration.set_put(
            maxXY=server.config.pos_limits[0],
            minZ=server.config.pos_limits[1],
            maxZ=server.config.pos_limits[2],
            maxUV=server.config.pos_limits[3],
            minW=server.config.pos_limits[4],
            maxW=server.config.pos_limits[5],
            maxVelocityXY=server.config.vel_limits[0],
            maxVelocityUV=server.config.vel_limits[1],
            maxVelocityZ=server.config.vel_limits[2],
            maxVelocityW=server.config.vel_limits[3],
            initialX=server.config.initial_pos[0],
            initialY=server.config.initial_pos[1],
            initialZ=server.config.initial_pos[2],
            initialU=server.config.initial_pos[3],
            initialV=server.config.initial_pos[4],
            initialW=server.config.initial_pos[5],
            pivotX=server.config.pivot[0],
            pivotY=server.config.pivot[1],
            pivotZ=server.config.pivot[2],
            maxDisplacementStrut=server.config.strut_displacement_max,
            maxVelocityStrut=server.config.strut_velocity_max,
            accelerationStrut=server.config.strut_acceleration,
        )

    def telemetry_callback(self, server):
        """Called when the TCP/IP controller outputs telemetry.

        Parameters
        ----------
        server : `lsst.ts.hexrotcomm.CommandTelemetryServer`
            TCP/IP server.
        """
        self.evt_summaryState.set_put(summaryState=self.summary_state)
        # Strangely telemetry.state, offline_substate and enabled_substate
        # are all floats from the controller. But they should only have
        # integer value, so I output them as integers.
        self.evt_controllerState.set_put(
            controllerState=int(server.telemetry.state),
            offlineSubstate=int(server.telemetry.offline_substate),
            enabledSubstate=int(server.telemetry.enabled_substate),
            applicationStatus=server.telemetry.application_status,
        )

        pos_error = [
            server.telemetry.measured_pos[i] - server.telemetry.commanded_pos[i]
            for i in range(6)
        ]
        self.tel_actuators.set_put(
            calibrated=server.telemetry.strut_encoder_microns,
            raw=server.telemetry.strut_encoder_raw,
        )
        self.tel_application.set_put(
            demand=server.telemetry.commanded_pos,
            position=server.telemetry.measured_pos,
            error=pos_error,
        )
        self.tel_electrical.set_put(
            copleyStatusWordDrive=server.telemetry.status_word,
            copleyLatchingFaultStatus=server.telemetry.latching_fault_status_register,
        )

        actuator_in_position = tuple(
            status & Hexapod.ApplicationStatus.HEX_MOVE_COMPLETE_MASK
            for status in server.telemetry.application_status
        )
        self.evt_actuatorInPosition.set_put(inPosition=actuator_in_position)
        self.evt_inPosition.set_put(inPosition=all(actuator_in_position))

        self.evt_commandableByDDS.set_put(
            state=bool(
                server.telemetry.application_status[0]
                & Hexapod.ApplicationStatus.DDS_COMMAND_SOURCE
            )
        )

        safety_interlock = (
            server.telemetry.application_status[0]
            & Hexapod.ApplicationStatus.SAFTEY_INTERLOCK
        )
        self.evt_interlock.set_put(
            detail="Engaged" if safety_interlock else "Disengaged"
        )

    def make_mock_controller(self, initial_ctrl_state):
        return mock_controller.MockMTHexapodController(
            log=self.log,
            index=self.salinfo.index,
            initial_state=initial_ctrl_state,
            command_port=self.server.command_port,
            telemetry_port=self.server.telemetry_port,
        )

    def _make_position_set_command(self, data):
        """Make a POSITION_SET command for the low-level controller.

        Parameters
        ----------
        data : ``struct with x, y, z, u, v, w`` fields.
            Data from the ``move`` or ``compensatedMove`` command.
            May also be data from the ``offset`` command,
            but the fields must be changed to absolute positions.
        """
        utils.check_symmetrical_range(
            data.x,
            "x",
            self.server.config.pos_limits[0],
            ExceptionClass=salobj.ExpectedError,
        )
        utils.check_symmetrical_range(
            data.y,
            "y",
            self.server.config.pos_limits[0],
            ExceptionClass=salobj.ExpectedError,
        )
        utils.check_range(
            data.z,
            "z",
            self.server.config.pos_limits[1],
            self.server.config.pos_limits[2],
            ExceptionClass=salobj.ExpectedError,
        )
        utils.check_symmetrical_range(
            data.u,
            "u",
            self.server.config.pos_limits[3],
            ExceptionClass=salobj.ExpectedError,
        )
        utils.check_symmetrical_range(
            data.v,
            "v",
            self.server.config.pos_limits[3],
            ExceptionClass=salobj.ExpectedError,
        )
        utils.check_range(
            data.w,
            "w",
            self.server.config.pos_limits[4],
            self.server.config.pos_limits[5],
            ExceptionClass=salobj.ExpectedError,
        )
        return self.make_command(
            code=enums.CommandCode.POSITION_SET,
            param1=data.x,
            param2=data.y,
            param3=data.z,
            param4=data.u,
            param5=data.v,
            param6=data.w,
        )

    def load_luts(self, elevation_prefix, azimuth_prefix, temperature_prefix):
        """Load compensation lookup tables.

        Parameters
        ----------
        elevation_prefix : `str`
            Elevation compensation lookup table file prefix.
            For example "default" will load "default_elevation_lut.dat"
            in self.compensation_data_dir.
            If blank then the existing LUT data is left unchanged.
        azimuth_prefix : `str`
            Elevation compensation lookup table file prefix.
            If blank then the existing LUT data is left unchanged.
        temperature_prefix : `str`
            Temperature compensation lookup table file prefix.
            If blank then the existing LUT data is left unchanged.
        """
        if elevation_prefix:
            self.elevation_lut = LookupTable(
                self.compensation_data_dir / f"{elevation_prefix}_elevation_lut.dat"
            )
        if azimuth_prefix:
            self.azimuth_lut = LookupTable(
                self.compensation_data_dir / f"{azimuth_prefix}_azimuth_lut.dat"
            )
        if temperature_prefix:
            self.temperature_lut = LookupTable(
                self.compensation_data_dir / f"{temperature_prefix}_temperature_lut.dat"
            )

    def get_compensation_data_dir(self, index):
        """Get the standard directory containing compensation data files.

        These files are in the ``ts_config_mttcs`` package,
        in ``Hexapod/v1``, in a subdirectory whose name
        depends on the CSC index.
        """
        config_env_var_name = "TS_CONFIG_MTTCS_DIR"
        try:
            config_pkg_dir = os.environ[config_env_var_name]
        except KeyError:
            raise RuntimeError(
                f"Environment variable {config_env_var_name} not defined"
            )
        index_subdir = {
            enums.SalIndex.CAM_HEXAPOD: "Camera",
            enums.SalIndex.M2_HEXAPOD: "M2",
        }[index]
        compensation_data_dir = (
            pathlib.Path(config_pkg_dir).resolve() / "Hexapod" / "v1" / index_subdir
        )
        if not compensation_data_dir.is_dir():
            raise RuntimeError(
                f"{compensation_data_dir!r} does not exist or is not a directory"
            )
        return compensation_data_dir

    def _make_offset_set_command(self, data):
        """Make a POSITION_SET offset command for the low-level controller
        using data from the offset CSC command.
        """
        position_data = copy.copy(data)
        position_data.x += self.server.telemetry.commanded_pos[0]
        position_data.y += self.server.telemetry.commanded_pos[1]
        position_data.z += self.server.telemetry.commanded_pos[2]
        position_data.u += self.server.telemetry.commanded_pos[3]
        position_data.v += self.server.telemetry.commanded_pos[4]
        position_data.w += self.server.telemetry.commanded_pos[5]

        return self._make_position_set_command(position_data)

    @classmethod
    async def amain(cls):
        """Make a CSC from command-line arguments and run it.
        """
        parser = argparse.ArgumentParser(f"Run {cls.__name__}")
        parser.add_argument("index", type=int, help="Hexapod index; 1=Camera 2=M2")
        parser.add_argument(
            "-s", "--simulate", action="store_true", help="Run in simulation mode?"
        )

        args = parser.parse_args()
        csc = cls(index=args.index, simulation_mode=int(args.simulate))
        await csc.done_task
