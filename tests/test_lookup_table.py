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

import pathlib
import unittest

import numpy as np

from lsst.ts import hexapod


class LookupTableTestCase(unittest.TestCase):
    def setUp(self):
        self.data_dir = pathlib.Path(__file__).resolve().parent / "data"

    def test_valid_input_names(self):
        for input_name in ("elevation", "azimuth", "temperature"):
            path = self.data_dir / f"{input_name}_lut.dat"
            lut = hexapod.LookupTable(path)
            self.assertEqual(lut.input_name, input_name)

    def test_constructor_errors(self):
        for name in (
            "bad_corr_name",
            "bad_input_name",
            "extra_column",
            "extra_value",
            "input_not_first",
            "missing_column",
            "missing_value",
            "non_monotonic_input_data",
            "wrong_order",
        ):
            path_to_bad_file = self.data_dir / f"lut_{name}.dat"
            with self.subTest(path_to_bad_file=path_to_bad_file):
                with self.assertRaises(RuntimeError):
                    hexapod.LookupTable(path_to_bad_file)

        with self.assertRaises(IOError):
            hexapod.LookupTable(self.data_dir / "non_existent_file.dat")

    def test_get_correction(self):
        path = self.data_dir / "elevation_lut.dat"
        lut = hexapod.LookupTable(path)
        self.assertEqual(len(lut.input_data), 5)
        self.assertEqual(len(lut.corr_data), 5)

        # Correction at input values should equal the associated
        # corr_data value.
        for i, input_value in enumerate(lut.input_data):
            corr = lut.get_correction(input_value)
            np.testing.assert_array_almost_equal(corr, lut.corr_data[i])

        # Check piecewise linear interpolation.
        for i, input_value0 in enumerate(lut.input_data[0:-1]):
            for frac in (0.1, 0.3, 0.7, 0.9):
                delta_input_value = lut.input_data[i + 1] - input_value0
                input_value = input_value0 + frac * delta_input_value
                corr0 = lut.corr_data[i]
                delta_corr = lut.corr_data[i + 1] - corr0
                desired_corr = corr0 + frac * delta_corr
                corr = lut.get_correction(input_value)
                np.testing.assert_array_almost_equal(corr, desired_corr)

        # Check invalid input values.
        for bad_input_value in (
            lut.input_data[0] - 0.0001,
            lut.input_data[0] - 100,
            lut.input_data[-1] + 0.0001,
            lut.input_data[-1] + 100,
        ):
            with self.assertRaises(ValueError):
                lut.get_correction(bad_input_value)


if __name__ == "__main__":
    unittest.main()
