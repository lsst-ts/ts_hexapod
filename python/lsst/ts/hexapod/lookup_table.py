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

__all__ = ["LookupTable"]

import bisect

import numpy as np


class LookupTable:
    """A linearly interpolated lookup table for hexapod corrections.

    Parameters
    ----------
    path : `pathlib.Path` or `str`
        Path to data file. See `Notes`_ for the data format.

    Attributes
    ----------
    input_data : `list` [`float`]
        List of input values.
    corr_data : `list` [`numpy.ndarray`]
        List of (x, y, z, u, v, w) correction at a given input value.
        The same length as ``input_data``.

    Raises
    ------
    IOError
        If the file cannot be read.
    RuntimeError
        If the file data format is incorrect.

    Notes
    -----
    The file format is an ASCII table of correction values as follows:
    * Empty lines and lines that begin with # (comments) are ignored.
    * Data is whitespace separated.
    * The first line of data is the column names, in order:

         input_name x y z u v w

      where input_name is one of "elevation", "azimuth", or "temperature".
    * Column names may have any case you like (e.g. elevation or Elevation).
    * All columns must be present and no extra columns are allowed.
    * Each subsequent line is data for one entry in the lookup table
      (7 whitespace-separated float values).
    * The input values must increase monotonically.

    Units are as follows:
    * azimuth, elevation, u, v, w: deg
    * temperature: Celscius
    * x, y, z: um

    For an example see ``tests/data/elevation_lut.dat``
    """

    def __init__(self, path):
        corr_names = ["x", "y", "z", "u", "v", "w"]
        input_data = []
        corr_data = []
        self.input_name = None
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    # Blank line: skip it
                    continue
                if line[0] == "#":
                    # Comment: skip it
                    continue
                entries = line.split()
                if len(entries) != 7:
                    raise RuntimeError(
                        f"Line {line} has {len(entries)} entries instead of 7"
                    )

                if self.input_name is None:
                    # This line is column names
                    input_name = entries[0].lower()
                    if input_name not in {"elevation", "azimuth", "temperature"}:
                        raise RuntimeError(
                            f"Input name {entries[0]!r} must be one of "
                            "elevation, azimuth, or temperature (case blind)"
                        )
                    corr_names = [name.lower() for name in entries[1:]]
                    if corr_names != ["x", "y", "z", "u", "v", "w"]:
                        raise RuntimeError(
                            f"Column names {entries[1:]} must be x, y, z, u, v, w (case blind)"
                        )
                    self.input_name = input_name
                else:
                    # This is a line of data
                    data = [float(entry) for entry in entries]
                    input_data.append(data[0])
                    corr_data.append(np.array(data[1:], dtype=float))

        if input_data != sorted(input_data):
            raise RuntimeError(
                f"{self.input_name!r} data not monotonically increasing: {input_data}"
            )

        self.input_data = input_data
        self.corr_data = corr_data

    def get_correction(self, input_value):
        """Get corrections for a given input value.

        Parameters
        ----------
        input_value : `float`
            Input value.

        Returns
        -------
        corrections : `np.array`
            Correction for x, y, z (um), u, v, w (deg),
            as a float numpy array of 6 elements.

        Raises
        ------
        ValueError
            If ``input_value`` < self.input_data[0] or > self.input_data[-1]
        """
        if input_value < self.input_data[0] or input_value > self.input_data[-1]:
            raise ValueError(
                f"input_value {input_value} out of range "
                f"[{self.input_data[0]}, {self.input_data[-1]}]"
            )
        ind = bisect.bisect_right(self.input_data, input_value) - 1
        if ind == len(self.input_data) - 1:
            ind -= 1
        frac_value = (input_value - self.input_data[ind]) / (
            self.input_data[ind + 1] - self.input_data[ind]
        )
        row = self.corr_data[ind]
        next_row = self.corr_data[ind + 1]
        corr = row + frac_value * (next_row - row)
        return corr
