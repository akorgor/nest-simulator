# -*- coding: utf-8 -*-
#
# hl_api_nodes.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

"""
Classes and functions for node handling
"""

import nest
from .hl_api_helper import *
from .hl_api_info import SetStatus


class GIDCollection(object):
    """
    Class for GIDCollection.

    GIDCollection represents the nodes of a network. The class supports
    iteration, concatination, indexing, slicing, membership, convertion to and
    from lists, test for membership, and test for equality.

    A GIDCollection is created by the ``Create`` function, or by converting a
    list of nodes to a GIDCollection with ``nest.GIDCollection(list)``.

    By iterating over the GIDCollection you get the gids. If you apply the
    ``items()`` function, you can also read the modelID.

    **Example**
        ::
            import nest

            nest.ResetKernel()

            # Create GIDCollection representing nodes
            gc = nest.Create('iaf_neuron', 10)

            # Print gids and modelID
            for gid, mid in gc.items():
                print(gid, mid)

            # Convert from list
            gids_in = [2, 4, 6, 8]
            new_gc = nest.GIDCollection(gids_in)

            # Concatination
            Enrns = nest.Create('aeif_cond_alpha', 600)
            Inrns = nest.Create('iaf_neuron', 400)
            nrns = Enrns + Inrns

            # Indexing, slicing and membership
            print(new_gc[2])
            print(new_gc[1:2])
            6 in new_gc
    """

    _datum = None

    def __init__(self, data):
        if isinstance(data, nest.SLIDatum):
            if data.dtype != "gidcollectiontype":
                raise TypeError("Need GIDCollection Datum.")
            self._datum = data
        else:
            # Data from user, must be converted to datum
            self._datum = nest.sli_func('cvd', data)

    def __iter__(self):
        # Naive implementation
        gc_as_list = nest.sli_func('cva', self._datum)
        try:
            it = iter(gc_as_list)
        except TypeError:
            raise TypeError("The GIDCollection needs to be converted to list \
                             in order to be iterable")
        return it

    def items(self):
        # Naive implementation
        gc_with_mid_as_list = nest.sli_func('cva_with_mid', self._datum)
        try:
            it = iter(gc_with_mid_as_list)
        except TypeError:
            raise TypeError("The GIDCollection needs to be converted to list \
                             in order to be iterable")
        return it

    def __add__(self, other):
        if not isinstance(other, GIDCollection):
            return NotImplemented
        return GIDCollection(nest.sli_func('join', self._datum, other._datum))

    def __getitem__(self, key):
        if isinstance(key, slice):
            return (GIDCollection(nest.sli_func('Take',
                    self._datum, [key.start, key.stop, key.step])))
        else:
            gid = nest.sli_func('get', self._datum, key)
            try:
                gid + 1
            except TypeError:
                raise TypeError("Slicing of a GIDCollection should return a \
                                gid")
            return gid

    def __contains__(self, gid):
        return nest.sli_func('MemberQ', self._datum, gid)

    def __eq__(self, other):
        if not isinstance(other, GIDCollection):
            return NotImplemented
        return self._datum == other._datum

    def __neq__(self, other):
        if not isinstance(other, GIDCollection):
            return NotImplemented
        return not self == other


@check_stack
def Create(model, n=1, params=None):
    """Create n instances of type model.

    Parameters
    ----------
    model : str
        Name of the model to create
    n : int, optional
        Number of instances to create
    params : TYPE, optional
        Parameters for the new nodes. A single dictionary or a list of
        dictionaries with size n. If omitted, the model's defaults are used.

    Returns
    -------
    list:
        Global IDs of created nodes
    """

    if isinstance(params, dict):
        cmd = "/%s 3 1 roll exch Create" % model
        sps(params)
    else:
        cmd = "/%s exch Create" % model

    sps(n)
    sr(cmd)

    # last_gid = spp()
    # gids = tuple(range(last_gid - n + 1, last_gid + 1))
    gids = GIDCollection(spp())

    if params is not None and not isinstance(params, dict):
        try:
            SetStatus(gids, params)
        except:
            warnings.warn(
                "SetStatus() call failed, but nodes have already been " +
                "created! The GIDs of the new nodes are: {0}.".format(gids))
            raise

    return gids


@check_stack
def GetLID(gid):
    """Return the local id of a node with the global ID gid.

    Parameters
    ----------
    gid : int
        Global id of node

    Returns
    -------
    int:
        Local id of node

    Raises
    ------
    NESTError
    """

    if len(gid) > 1:
        raise NESTError("GetLID() expects exactly one GID.")

    sps(gid[0])
    sr("GetLID")

    return spp()
