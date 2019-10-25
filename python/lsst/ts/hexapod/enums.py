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

__all__ = ["SalIndex", "CommandCode", "FrameId", "SetStateParam", "SetEnabledSubstateParam"]

import enum


class SalIndex(enum.IntEnum):
    CAM_HEXAPOD = 1
    M2_HEXAPOD = 2


class CommandCode(enum.IntEnum):
    """Codes for the cmd field of commands.

    In the Moog code these are defined in enum cmdType.
    I have reworded them for clarity.
    """
    SET_STATE = 0x8000
    SET_ENABLED_SUBSTATE = 0x8001
    POSITION_SET = 0x8004
    SET_PIVOTPOINT = 0x8007
    CONFIG_ACCEL = 0x800B
    CONFIG_VEL = 0x800C
    CONFIG_LIMITS = 0x800D
    OFFSET = 0x8010


class FrameId(enum.IntEnum):
    """Frame IDs for Camera and M2 hexapod telemetry and configuration.
    """
    CAM_TELEMETRY = 0x7
    CAM_CONFIG = 0x1B
    M2_TELEMETRY = 0x8
    M2_CONFIG = 0x1C


class SetStateParam(enum.IntEnum):
    """Values for ``command.param1`` when
    ``command.cmd = CommandCode.SET_STATE``.

    Called ``TriggerCmds`` in Moog code.
    """
    INVALID = 0
    START = enum.auto()
    ENABLE = enum.auto()
    STANDBY = enum.auto()
    DISABLE = enum.auto()
    EXIT = enum.auto()
    CLEAR_ERROR = enum.auto()
    ENTER_CONTROL = enum.auto()


class SetEnabledSubstateParam(enum.IntEnum):
    """Substates for the ENABLED state.
    """
    ENABLED_INVALID = 0
    MOVE_POINT_TO_POINT = enum.auto()
    TRACK = enum.auto()
    STOP = enum.auto()
    INITIALIZE = enum.auto()
    RELATIVE = enum.auto()
    CONST_VEL = enum.auto()
    SPARE2 = enum.auto()
    MOVE_LUT = enum.auto()