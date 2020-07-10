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

__all__ = ["HexapodCommander"]

import numpy as np

from lsst.ts import hexrotcomm

from . import enums

STD_TIMEOUT = 5  # timeout for command ack


def as_bool(value):
    """Return a string cast to a bool.

    The following are false: 0, f, false
    The following are true: 1, t, true
    No other values are valid.
    """
    return {"0": False, "f": False, "false": False, "1": True, "t": True, "true": True}[
        value
    ]


class HexapodCommander(hexrotcomm.CscCommander):
    """Command the Hexapod CSC from the command line.

    Parameters
    ----------
    index : `int`
        SAL index of Hexapod CSC.

    Read commands from stdin and write updated events and telemetry to stdout.
    The telemetry is filtered so that tiny changes due to encoder jitter
    are ignored.

    See bin/command_hexapod.py for an example of how to use this class.
    """

    def __init__(self, index):
        index = enums.SalIndex(index)
        super().__init__(
            name="Hexapod",
            index=index,
            help_text="""Special commands:
* exit  # Quit the interpreter.
* help  # Print this help.

State transitions commands (none take arguments):
* enterControl
* start
* enable
* disable
* standby
* exitControl
* clearError

Other commands and arguments:
* configureAcceleration acceleration                # Set strut acceleration: µm/s2
* configureLimits maxXY minZ maxZ maxUV minW maxW   # Set position limits: µm µm µm deg deg deg
* configureVelocity xy z uv w   # Set velocity: µm/s deg/s µm/s deg/s
* compensatedMove elevation azimuth temperature x y z u v w synch  # Move with compensation
* move x y z u v w synch        # Move to the specified position and orientation
* offset x y z u v w synch      # Offset by a specified change in position and orientation
* pivot x y z                   # Set the pivot point
* stop

Position (x y z) is in µm
Angles (u, v, w, elevation, and azimuth) are in degrees
sync is 1 to synchronize motion (actuators start and stop together), 0 to not
Temperature is in Celsius
"Compensation" means pivot position and orientation are compensated
for telescope position and ambient temperature.

For example:
  move 5 5 5 0.001 0 0 0
  stop  # in case you want to stop a move early
  exit""",
        )

    async def do_configureAcceleration(self, args):
        """Implement the configureAcceleration command.

        Parameters
        ----------
        args : `List` [`float`]
            One value: acceleration (deg/sec2).
        """
        kwargs = self.check_arguments(args, "acceleration")
        await self.remote.cmd_configureAcceleration.set_start(
            **kwargs, timeout=STD_TIMEOUT
        )

    async def do_configureLimits(self, args):
        """Implement the configureLimits command.

        Parameters
        ----------
        args : `List` [`float`]
            Six values: maxXY, minZ, maxZ (µm), maxUV, minW, maxW (deg).
        """
        kwargs = self.check_arguments(
            args, "maxXY", "minZ", "maxZ", "maxUV", "minW", "maxW"
        )
        await self.remote.cmd_configureLimits.set_start(**kwargs, timeout=STD_TIMEOUT)

    async def do_configureVelocity(self, args):
        """Implement the configureVelocity command.

        Parameters
        ----------
        args : `List` [`float`]
            Four values: xy (µm/sec), z (µm/sec), uv (deg/sec), w (deg/sec)
        """
        kwargs = self.check_arguments(args, "xy", "z", "uv", "w")
        await self.remote.cmd_configureVelocity.set_start(**kwargs, timeout=STD_TIMEOUT)

    async def do_move(self, args):
        """Implement the move command.

        Parameters
        ----------
        args : `List` [`float`]
            7 values: x, y, z (µm), u, v, w (deg), sync (bool).
        """
        self.check_arguments(args, "x", "y", "z", "u", "v", "w", ("sync", as_bool))
        await self.remote.cmd_move.start(timeout=STD_TIMEOUT)

    async def do_compensatedMove(self, args):
        """Implement the compensatedMove command.

        Parameters
        ----------
        args : `List` [`float`]
            10 values:
                elevation (deg), azimuth (deg), temperature (C),
                x, y, z (µm), u, v, w (deg), sync (bool).
        """
        kwargs = self.check_arguments(
            args,
            "elevation",
            "azimuth",
            "temperature",
            "x",
            "y",
            "z",
            "u",
            "v",
            "w",
            ("sync", as_bool),
        )
        await self.remote.cmd_compensatedMove.set_start(**kwargs, timeout=STD_TIMEOUT)

    async def do_offset(self, args):
        """Implement the offset command.

        Parameters
        ----------
        args : `List` [`float`]
            7 values: x, y, z (µm), u, v, w (deg), sync (bool).
        """
        kwargs = self.check_arguments(
            args, "x", "y", "z", "u", "v", "w", ("sync", as_bool)
        )
        await self.remote.cmd_offset.set_start(**kwargs, timeout=STD_TIMEOUT)

    async def do_pivot(self, args):
        """Implement the pivot command.

        Parameters
        ----------
        args : `List` [`float`]
            Three values: x, y, z (µm).
        """
        kwargs = self.check_arguments(args, "x", "y", "z")
        await self.remote.cmd_pivot.set_start(**kwargs, timeout=STD_TIMEOUT)

    async def do_stop(self, args):
        """Implement the stop command.

        Parameters
        ----------
        args : `List` [`float`]
            Ignored.
        """
        # Don't check arguments, just STOP.
        await self.remote.cmd_stop.start(timeout=STD_TIMEOUT)

    def positions_close(self, position1, position2):
        """Return True if two positions are nearly equal.

        Parameters
        ----------
        position1 : `List` [`float`]
            Position 1: x, y, z (µm) and rotx, roty, rotz (deg)
        position2 : `List` [`float`]
            Position 2: x, y, z (µm) and rotx, roty, rotz (deg)
        """
        return np.allclose(position1[:3], position2[:3], atol=1) and np.allclose(
            position1[3:], position2[3:], atol=1e-5
        )

    async def tel_actuators_callback(self, data):
        """Callback for Actuators telemetry.

        Output Actuators telemetry data if the values have changed enough
        to be interesting.

        Parameters
        ----------
        data : self.controller.tel_actuators.DataType.
            Actuators data.
        """
        if self.previous_tel_actuators is not None and np.allclose(
            self.previous_tel_actuators.calibrated, data.calibrated, atol=1
        ):
            return
        self.previous_tel_actuators = data
        print(f"actuators: {self.format_data(data)}")

    async def tel_application_callback(self, data):
        """Callback for the Application telemetry.

        Output Application telemetry if the values have changed enough
        to be interesting.

        Parameters
        ----------
        data : self.controller.tel_application.DataType.
            Actuators data.
        """
        if (
            self.previous_tel_application is not None
            and self.positions_close(
                self.previous_tel_application.position, data.position
            )
            and np.array_equal(data.demand, self.previous_tel_application.demand)
        ):
            return
        self.previous_tel_application = data
        print(f"application: {self.format_data(data)}")
