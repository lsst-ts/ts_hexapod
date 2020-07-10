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

import ctypes


ELEVATION_ELEMENTS = 19
AZIMUTH_ELEMENTS = 37
TEMPERATURE_ELEMENTS = 9

# What are these for?
CAM_ERROR_CODE = 6201
M2_ERROR_CODE = 6151

# What is this?
HEXAPOD_DDS_TLM_CNT = 392


class Config(ctypes.Structure):
    """Hexapod configuration.

    In the low-level controller code these are defined in
    ``ddsConfigTelemetryStreamStructure_t``.
    """

    _pack_ = 1
    _fields_ = [
        ("strut_acceleration", ctypes.c_double),
        # Order: xy, zmin, max, uv, wmin, wmax
        ("pos_limits", ctypes.c_double * 6),
        # Order: xy, xy rotation, z, z rotation
        ("vel_limits", ctypes.c_double * 4),
        ("initial_pos", ctypes.c_double * 6),
        ("pivot", ctypes.c_double * 3),
        ("el_lut_index", ctypes.c_double * ELEVATION_ELEMENTS),
        ("el_lut_x", ctypes.c_double * ELEVATION_ELEMENTS),
        ("el_lut_y", ctypes.c_double * ELEVATION_ELEMENTS),
        ("el_lut_z", ctypes.c_double * ELEVATION_ELEMENTS),
        ("el_lut_u", ctypes.c_double * ELEVATION_ELEMENTS),
        ("el_lut_v", ctypes.c_double * ELEVATION_ELEMENTS),
        ("el_lut_w", ctypes.c_double * ELEVATION_ELEMENTS),
        ("az_lut_index", ctypes.c_double * AZIMUTH_ELEMENTS),
        ("az_lut_x", ctypes.c_double * AZIMUTH_ELEMENTS),
        ("az_lut_y", ctypes.c_double * AZIMUTH_ELEMENTS),
        ("az_lut_z", ctypes.c_double * AZIMUTH_ELEMENTS),
        ("az_lut_u", ctypes.c_double * AZIMUTH_ELEMENTS),
        ("az_lut_v", ctypes.c_double * AZIMUTH_ELEMENTS),
        ("az_lut_w", ctypes.c_double * AZIMUTH_ELEMENTS),
        ("temp_lut_index", ctypes.c_double * TEMPERATURE_ELEMENTS),
        ("temp_lut_x", ctypes.c_double * TEMPERATURE_ELEMENTS),
        ("temp_lut_y", ctypes.c_double * TEMPERATURE_ELEMENTS),
        ("temp_lut_z", ctypes.c_double * TEMPERATURE_ELEMENTS),
        ("temp_lut_u", ctypes.c_double * TEMPERATURE_ELEMENTS),
        ("temp_lut_v", ctypes.c_double * TEMPERATURE_ELEMENTS),
        ("temp_lut_w", ctypes.c_double * TEMPERATURE_ELEMENTS),
        ("strut_displacement_max", ctypes.c_double),
        ("strut_velocity_max", ctypes.c_double),
    ]
    # Set the frame ID when constructing the CSC
    FRAME_ID = None


class Telemetry(ctypes.Structure):
    """Hexapod telemetry.

    In the low-level controller code these are defined in
    ``ddsTelemetryStreamStructure_t``.
    """

    _pack_ = 1
    _fields_ = [
        ("status_word", ctypes.c_uint16 * 6),
        ("latching_fault_status_register", ctypes.c_uint16 * 6),
        ("copley_fault_status_register", ctypes.c_uint32 * 6),
        ("application_status", ctypes.c_uint32 * 6),
        ("input_pin_states", ctypes.c_uint32 * 3),
        # simulink telemetry
        ("state", ctypes.c_double),
        ("enabled_substate", ctypes.c_double),
        ("offline_substate", ctypes.c_double),
        ("test_state", ctypes.c_double),
        ("strut_encoder_raw", ctypes.c_double * 6),
        ("strut_encoder_microns", ctypes.c_double * 6),
        # commanded displacement (microns) and rotation (deg)
        ("commanded_pos", ctypes.c_double * 6),
        # measured displacement (microns) and rotation (deg)
        ("measured_pos", ctypes.c_double * 6),
        # commanded strut lengths (microns)
        ("commanded_length", ctypes.c_double * 6),
    ]
    # Set the frame ID when constructing the CSC
    FRAME_ID = None
