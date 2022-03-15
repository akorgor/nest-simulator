/*
 *  propagator_stability.h
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef PROPAGATOR_STABILITY_H
#define PROPAGATOR_STABILITY_H

// Propagators to handle similar tau_m and tau_syn_* time constants.
// For details, please see doc/userdoc/model_details/IAF_neurons_singularity.ipynb.

class propagator
{
public:

  propagator();

  void calculate_constants( double tau_syn, double tau, double c );
  double propagator_31( double tau_syn, double tau, double C, double h ) const;
  double propagator_32( double tau_syn, double tau, double C, double h ) const;

private:
  double alpha_;
  double beta_;
  double gamma_;    //!< 1/c * 1/(1/tau_syn - 1/tau)
  double gamma_sq_; //!< 1/c * 1/(1/tau_syn - 1/tau)^2
};

double propagator_31( double tau_syn, double tau, double C, double h );
double propagator_32( double tau_syn, double tau, double C, double h );

#endif
