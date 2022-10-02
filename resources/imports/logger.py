#!/usr/bin/env python
# ******************************************************************************
# Copyright © 2022 Johan Mejia (josmejiam@correo.udistrital.edu.co)
#
# This file is part of PyRat-RL (see https://github.com/Johansmm/PyRat-RL).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# ******************************************************************************
"""Generate a custom logger."""
import os
import logging


__all__ = ["logger"]


def _get_log_level():
    """Get log level from enviroment variable ``LOG_LEVEL``.
    If it is not defined, used default level (``NOTSET``)

    Returns
    -------
    int
        Logger level

    Raises
    ------
    ValueError
        Input logger name is not recognized
    """
    loglevel = os.environ.get("LOG_LEVEL", "NOTSET")
    if loglevel.isnumeric():
        loglevel = logging.getLevelName(eval(loglevel))
    loglevel_att = getattr(logging, loglevel.upper(), None)
    if loglevel_att is None:
        raise ValueError(f"Impossible to get the LOG_LEVEL of {loglevel}")
    return loglevel_att


def _create_logger():
    """Create a dedicated logger for general purposes

    Returns
    -------
    logging.Logger
        Object to logger custom messages
    """
    # Append level alias based on the position
    for idx, level in enumerate(list(logging._nameToLevel.keys())[::-1]):
        logging.addLevelName(idx, level)
    # Create a specific logger, and set level provided by enviroment
    logger = logging.getLogger('pyrat')
    logger.setLevel(_get_log_level())
    # Add a handler to set a specific format
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s][%(asctime)s][%(filename)s] %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = _create_logger()
