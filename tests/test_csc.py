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

import asyncio
import unittest
import time

import asynctest
import numpy as np

from lsst.ts import salobj
from lsst.ts import hexapod
from lsst.ts import hexrotcomm
from lsst.ts.idl.enums import Hexapod

STD_TIMEOUT = 5  # timeout for command ack

index_gen = salobj.index_generator(imin=1, imax=2)


class TestHexapodCsc(hexrotcomm.BaseCscTestCase, asynctest.TestCase):

    def basic_make_csc(self, initial_state, simulation_mode=1):
        return hexapod.HexapodCsc(index=next(index_gen),
                                  initial_state=initial_state,
                                  simulation_mode=simulation_mode)

    async def test_bin_script(self):
        """Test running from the command line script.
        """
        await self.check_bin_script(name="Hexapod",
                                    index=next(index_gen),
                                    exe_name="run_hexapod.py")

    async def test_standard_state_transitions(self):
        enabled_commands = ("configureVelocity", "configureAcceleration", "configureLimits",
                            "move", "moveLUT", "offset", "offsetLUT", "pivot", "stop")
        await self.check_standard_state_transitions(enabled_commands=enabled_commands)

    async def test_configure_acceleration(self):
        """Test the configureAcceleration command.
        """
        await self.make_csc(initial_state=salobj.State.ENABLED)
        data = await self.remote.evt_configuration.next(flush=False, timeout=STD_TIMEOUT)
        initial_limit = data.accelerationAccmax
        new_limit = initial_limit - 0.1
        await self.remote.cmd_configureAcceleration.set_start(accmax=new_limit, timeout=STD_TIMEOUT)
        data = await self.remote.evt_configuration.next(flush=False, timeout=STD_TIMEOUT)
        self.assertAlmostEqual(data.accelerationAccmax, new_limit)

        for bad_accmax in (-1, 0, hexapod.MAX_ACCEL_LIMIT + 0.001):
            with self.subTest(bad_accmax=bad_accmax):
                with salobj.assertRaisesAckError(ack=salobj.SalRetCode.CMD_FAILED):
                    await self.remote.cmd_configureAcceleration.set_start(accmax=bad_accmax,
                                                                          timeout=STD_TIMEOUT)

    async def test_configure_limits(self):
        """Test the configureLimits command.
        """
        def get_limits(data):
            """Get position limits from a configuration sample."""
            return (data.limitXYMax, data.limitZMin, data.limitZMax,
                    data.limitUVMax, data.limitWMin, data.limitWMax)

        await self.make_csc(initial_state=salobj.State.ENABLED)
        await self.assert_next_controller_state(controllerState=Hexapod.ControllerState.ENABLED,
                                                enabledSubstate=Hexapod.EnabledSubstate.STATIONARY)

        data = await self.remote.evt_configuration.next(flush=False, timeout=STD_TIMEOUT)
        initial_limits = get_limits(data)

        new_limits = tuple(lim*0.1 for lim in initial_limits)
        await self.remote.cmd_configureLimits.set_start(xymax=new_limits[0],
                                                        zmin=new_limits[1],
                                                        zmax=new_limits[2],
                                                        uvmax=new_limits[3],
                                                        wmin=new_limits[4],
                                                        wmax=new_limits[5],
                                                        timeout=STD_TIMEOUT)
        data = await self.remote.evt_configuration.next(flush=False, timeout=STD_TIMEOUT)
        reported_limits = get_limits(data)
        for i in range(4):
            self.assertAlmostEqual(new_limits[i], reported_limits[i])

        # Make sure we cannot move to a position outside the new limits
        good_max_position = self.limits_to_max_position(new_limits)
        good_min_position = self.limits_to_min_position(new_limits)

        bad_max_position = tuple(val*1.01 for val in good_max_position)
        bad_min_position = tuple(val*1.01 for val in good_min_position)
        move_kwargs = self.make_xyzuvw_kwargs(bad_max_position)
        with salobj.assertRaisesAckError():
            await self.remote.cmd_move.set_start(**move_kwargs, timeout=STD_TIMEOUT)
        move_kwargs = self.make_xyzuvw_kwargs(bad_min_position)
        with salobj.assertRaisesAckError():
            await self.remote.cmd_move.set_start(**move_kwargs, timeout=STD_TIMEOUT)

        # Make sure we can move to the new limits
        await self.basic_check_move(destination=good_max_position,
                                    est_move_duration=2,
                                    elaztemp=None)

        await self.basic_check_move(destination=good_min_position,
                                    est_move_duration=2,
                                    elaztemp=None)

        def make_modified_limits(i, value):
            """Make modified limits from initial_limits, setting one element
            to a new value.
            """
            bad_limits = list(initial_limits)
            bad_limits[i] = value
            return tuple(bad_limits)

        for bad_limits in (
            make_modified_limits(0, 0),
            make_modified_limits(0, self.csc.xy_max_limit + 0.001),
            make_modified_limits(1, self.csc.z_min_limit - 0.001),
            make_modified_limits(2, self.csc.z_max_limit + 0.001),
            make_modified_limits(3, 0),
            make_modified_limits(3, self.csc.uv_max_limit + 0.001),
            make_modified_limits(4, self.csc.w_min_limit - 0.001),
            make_modified_limits(5, self.csc.w_max_limit + 0.001),
        ):
            with self.subTest(bad_limits=bad_limits):
                with salobj.assertRaisesAckError(ack=salobj.SalRetCode.CMD_FAILED):
                    await self.remote.cmd_configureLimits.set_start(xymax=bad_limits[0],
                                                                    zmin=bad_limits[1],
                                                                    zmax=bad_limits[2],
                                                                    uvmax=bad_limits[3],
                                                                    wmin=bad_limits[4],
                                                                    wmax=bad_limits[5],
                                                                    timeout=STD_TIMEOUT)

    async def test_configure_velocity(self):
        """Test the configureVelocity command.
        """
        def get_velocity_limits(data):
            """Get the velocity limits from a configuration sample."""
            return (data.velocityXYMax, data.velocityRxRyMax, data.velocityZMax, data.velocityRzMax)

        await self.make_csc(initial_state=salobj.State.ENABLED)
        data = await self.remote.evt_configuration.next(flush=False, timeout=STD_TIMEOUT)
        initial_vel_limits = get_velocity_limits(data)
        new_vel_limits = tuple(lim - 0.01 for lim in initial_vel_limits)
        await self.remote.cmd_configureVelocity.set_start(xymax=new_vel_limits[0],
                                                          rxrymax=new_vel_limits[1],
                                                          zmax=new_vel_limits[2],
                                                          rzmax=new_vel_limits[3],
                                                          timeout=STD_TIMEOUT)
        data = await self.remote.evt_configuration.next(flush=False, timeout=STD_TIMEOUT)
        reported_limits = get_velocity_limits(data)
        for i in range(4):
            self.assertAlmostEqual(new_vel_limits[i], reported_limits[i])

        bad_linear_vel_limit = hexapod.MAX_LINEAR_VEL_LIMIT + 0.001
        bad_angular_vel_limit = hexapod.MAX_ANGULAR_VEL_LIMIT + 0.001
        for bad_vel_limits in (
            (0, initial_vel_limits[1], initial_vel_limits[2], initial_vel_limits[3]),
            (bad_linear_vel_limit, initial_vel_limits[1], initial_vel_limits[2], initial_vel_limits[3]),
            (initial_vel_limits[0], 0, initial_vel_limits[2], initial_vel_limits[3]),
            (initial_vel_limits[0], bad_angular_vel_limit, initial_vel_limits[2], initial_vel_limits[3]),
            (initial_vel_limits[0], initial_vel_limits[1], 0, initial_vel_limits[3]),
            (initial_vel_limits[0], initial_vel_limits[1], bad_linear_vel_limit, initial_vel_limits[3]),
            (initial_vel_limits[0], initial_vel_limits[1], initial_vel_limits[2], 0),
            (initial_vel_limits[0], initial_vel_limits[1], initial_vel_limits[2], bad_angular_vel_limit),
            (0, 0, 0, 0),
            (bad_linear_vel_limit, bad_angular_vel_limit, bad_linear_vel_limit, bad_angular_vel_limit),
        ):
            with self.subTest(bad_vel_limits=bad_vel_limits):
                with salobj.assertRaisesAckError(ack=salobj.SalRetCode.CMD_FAILED):
                    await self.remote.cmd_configureVelocity.set_start(xymax=bad_vel_limits[0],
                                                                      rxrymax=bad_vel_limits[1],
                                                                      zmax=bad_vel_limits[2],
                                                                      rzmax=bad_vel_limits[3],
                                                                      timeout=STD_TIMEOUT)

    async def test_move(self):
        await self.check_move(destination=(300, 400, -300, 0.01, 0.02, -0.015),
                              est_move_duration=1,
                              elaztemp=None)

    async def test_move_lut(self):
        await self.check_move(destination=(500, -300, 200, 0.03, -0.02, 0.03),
                              est_move_duration=1,
                              elaztemp=(32, 44, 15))

    async def check_next_position(self, desired_position):
        """Wait for Application telemetry and check the position.

        Parameters
        ----------
        desired_position : `List` [`float`]
            Desired position: x, y, z (µm), rotx, roty, rotz (deg)
        """
        data = await self.remote.tel_application.next(flush=True, timeout=STD_TIMEOUT)
        np.testing.assert_allclose(data.demand, desired_position)
        # Add slop to accommodate jitter added by the mock controller.
        np.testing.assert_allclose(data.position[:3], desired_position[:3], atol=1)
        np.testing.assert_allclose(data.position[3:], desired_position[3:], atol=1e-5)

    async def check_move(self, destination, est_move_duration, elaztemp):
        """Test point to point motion using the positionSet and move
        or moveLUT commands.

        Create the CSC and assume it starts with inPosition=False.

        Parameters
        ----------
        destination : `List` [`float`]
            Destination x, y, z (µm), rotx, roty, rotz (deg)
        est_move_duration : `float`
            Estimated move duration (sec)
        elaztemp : `List` [`float`] or `None`
            Elevation, azimuth (deg) and temperature (C)
            for the moveLUT command.
            If None then call move instead of moveLUT.
        """
        self.assertEqual(len(destination), 6)
        await self.make_csc(initial_state=salobj.State.ENABLED)
        await self.assert_next_controller_state(controllerState=Hexapod.ControllerState.ENABLED,
                                                enabledSubstate=Hexapod.EnabledSubstate.STATIONARY)
        await self.check_next_position(desired_position=(0,)*6)

        await self.basic_check_move(destination=destination,
                                    est_move_duration=est_move_duration,
                                    elaztemp=elaztemp)

    async def basic_check_move(self, destination, est_move_duration, elaztemp):
        """Test point to point motion using the positionSet and move
        or moveLUT commands.

        Unlike `check_move` this assumes the CSC has been created.
        If you have just created the CSC then be sure not to
        read the ``inPosition`` and ``actuatorInPosition`` events,
        so this code can check that they go false when the move starts.

        Parameters
        ----------
        destination : `List` [`float`]
            Destination x, y, z (µm), rotx, roty, rotz (deg)
        est_move_duration : `float`
            Estimated move duration (sec)
        elaztemp : `List` [`float`] or `None`
            Elevation, azimuth (deg) and temperature (C)
            for the moveLUT command.
            If None then call move instead of moveLUT.
        """
        t0 = time.time()
        move_kwargs = self.make_xyzuvw_kwargs(destination)
        if elaztemp is None:
            await self.remote.cmd_move.set_start(**move_kwargs,
                                                 timeout=STD_TIMEOUT)
        else:
            await self.remote.cmd_moveLUT.set_start(elevation=elaztemp[0],
                                                    azimuth=elaztemp[1],
                                                    temperature=elaztemp[2],
                                                    **move_kwargs,
                                                    timeout=STD_TIMEOUT)
        await self.assert_next_controller_state(
            controllerState=Hexapod.ControllerState.ENABLED,
            enabledSubstate=Hexapod.EnabledSubstate.MOVING_POINT_TO_POINT)
        await self.assert_next_controller_state(
            controllerState=Hexapod.ControllerState.ENABLED,
            enabledSubstate=Hexapod.EnabledSubstate.STATIONARY,
            timeout=STD_TIMEOUT+est_move_duration)
        data = await self.remote.evt_inPosition.next(flush=False, timeout=STD_TIMEOUT)
        self.assertFalse(data.inPosition)
        data = await self.remote.evt_inPosition.next(flush=False, timeout=STD_TIMEOUT)
        self.assertTrue(data.inPosition)
        data = await self.remote.evt_actuatorInPosition.next(flush=False, timeout=STD_TIMEOUT)
        self.assertEqual(tuple(data.inPosition), (False,)*6)
        # Check that actuatorInPosition returns all in position;
        # this should occur within 6 events (fewer if several actuators
        # finish their move at the same time).
        for i in range(6):
            try:
                data = await self.remote.evt_actuatorInPosition.next(flush=False, timeout=STD_TIMEOUT)
                self.assertIn(True, data.inPosition)
                if False not in data.inPosition:
                    break
            except asyncio.TimeoutError:
                self.fail("actuatorInPosition timed out before all True")
        else:
            self.fail("actuatorInPosition output 6 times, but not all True")

        print(f"Move duration: {time.time() - t0:0.2f} seconds")
        await self.check_next_position(desired_position=destination)

    async def test_offset(self):
        first_destination = (100, 200, -300, 0.01, 0.02, -0.015)
        offset = (50, -100, 135, 0.005, -0.005, 0.01)
        await self.check_offset(first_destination=first_destination,
                                offset=offset,
                                est_move_duration=1,
                                elaztemp=None)

    async def test_offset_lut(self):
        first_destination = (120, -180, 300, 0.01, 0.02, -0.015)
        offset = (-150, 100, -135, 0.005, -0.005, 0.01)
        elaztemp = (43, 56, 20)
        await self.check_offset(first_destination=first_destination,
                                offset=offset,
                                est_move_duration=1,
                                elaztemp=elaztemp)

    async def check_offset(self, first_destination, offset, est_move_duration, elaztemp):
        await self.check_move(destination=first_destination,
                              est_move_duration=1,
                              elaztemp=elaztemp)
        offset = (50, -100, 135, 0.005, -0.005, 0.01)
        desired_destination = np.add(first_destination, offset)
        offset_kwargs = self.make_xyzuvw_kwargs(offset)
        if elaztemp is None:
            await self.remote.cmd_offset.set_start(**offset_kwargs, timeout=STD_TIMEOUT)
        else:
            await self.remote.cmd_offsetLUT.set_start(**offset_kwargs,
                                                      elevation=elaztemp[0],
                                                      azimuth=elaztemp[1],
                                                      temperature=elaztemp[2],
                                                      timeout=STD_TIMEOUT)
        await self.assert_next_controller_state(
            controllerState=Hexapod.ControllerState.ENABLED,
            enabledSubstate=Hexapod.EnabledSubstate.MOVING_POINT_TO_POINT)
        await self.assert_next_controller_state(
            controllerState=Hexapod.ControllerState.ENABLED,
            enabledSubstate=Hexapod.EnabledSubstate.STATIONARY)
        await self.check_next_position(desired_position=desired_destination)

    async def test_stop_move(self):
        """Test stopping a point to point move.
        """
        # Command a move that moves all actuators equally
        destination = (0, 0, 1000, 0, 0, 0)
        await self.make_csc(initial_state=salobj.State.ENABLED)
        await self.assert_next_controller_state(controllerState=Hexapod.ControllerState.ENABLED,
                                                enabledSubstate=Hexapod.EnabledSubstate.STATIONARY)
        await self.check_next_position(desired_position=(0,)*6)
        move_kwargs = self.make_xyzuvw_kwargs(destination)
        await self.remote.cmd_move.set_start(**move_kwargs,
                                             timeout=STD_TIMEOUT)
        cmd_lengths = [actuator.end_pos for actuator in self.csc.mock_ctrl.hexapod.actuators]
        await self.assert_next_controller_state(
            controllerState=Hexapod.ControllerState.ENABLED,
            enabledSubstate=Hexapod.EnabledSubstate.MOVING_POINT_TO_POINT)
        await self.remote.cmd_stop.start(timeout=STD_TIMEOUT)
        await self.assert_next_controller_state(controllerState=Hexapod.ControllerState.ENABLED,
                                                enabledSubstate=Hexapod.EnabledSubstate.STATIONARY)
        await self.remote.tel_application.next(flush=True, timeout=STD_TIMEOUT)
        # The Mock controller does not compute position as a function
        # of actuator lengths, so test that motion halted by examining
        # the actuators.
        stopped_lengths = [actuator.end_pos for actuator in self.csc.mock_ctrl.hexapod.actuators]
        for i in range(6):
            self.assertNotAlmostEqual(cmd_lengths[i], stopped_lengths[i])

    async def test_pivot(self):
        """Test the pivot command.
        """
        await self.make_csc(initial_state=salobj.State.ENABLED)

    def make_xyzuvw_kwargs(self, data, sync=1):
        """Make a dict of x,y,z,u,v,w,sync based on from a list of positions.

        Parameters
        ----------
        data : `List` [`float`]
            x, y, z (µm), u, v, w (deg) data
        sync : `int` (optional)
            Should this be a synchronized move?
            Usually doesn't matter because the mock controller ignores it.
        """
        self.assertEqual(len(data), 6)
        return dict(x=data[0],
                    y=data[1],
                    z=data[2],
                    u=data[3],
                    v=data[4],
                    w=data[5],
                    sync=sync)

    def limits_to_max_position(self, limits):
        """Return the position corresponding to the maximum position limit.
        """
        return (limits[0],
                limits[0],
                limits[2],
                limits[3],
                limits[3],
                limits[5])

    def limits_to_min_position(self, limits):
        """Return the position corresponding to the minimum position limit.
        """
        return (-limits[0],
                -limits[0],
                limits[1],
                -limits[3],
                -limits[3],
                limits[4])


if __name__ == "__main__":
    unittest.main()
