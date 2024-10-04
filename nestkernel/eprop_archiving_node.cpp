/*
 *  eprop_archiving_node.cpp
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

// nestkernel
#include "eprop_archiving_node.h"
#include "eprop_archiving_node_impl.h"
#include "kernel_manager.h"

// sli
#include "dictutils.h"

namespace nest
{

std::map< std::string, EpropArchivingNodeRecurrent::surrogate_gradient_function >
  EpropArchivingNodeRecurrent::surrogate_gradient_funcs_ = {
    { "piecewise_linear", &EpropArchivingNodeRecurrent::compute_piecewise_linear_surrogate_gradient },
    { "exponential", &EpropArchivingNodeRecurrent::compute_exponential_surrogate_gradient },
    { "fast_sigmoid_derivative", &EpropArchivingNodeRecurrent::compute_fast_sigmoid_derivative_surrogate_gradient },
    { "arctan", &EpropArchivingNodeRecurrent::compute_arctan_surrogate_gradient }
  };


EpropArchivingNodeRecurrent::EpropArchivingNodeRecurrent()
  : EpropArchivingNode()
  , firing_rate_reg_( 0.0 )
  , f_av_( 0.0 )
  , n_spikes_( 0 )
{
}

EpropArchivingNodeRecurrent::EpropArchivingNodeRecurrent( const EpropArchivingNodeRecurrent& n )
  : EpropArchivingNode( n )
  , firing_rate_reg_( n.firing_rate_reg_ )
  , f_av_( n.f_av_ )
  , n_spikes_( n.n_spikes_ )
{
}

EpropArchivingNodeRecurrent::surrogate_gradient_function
EpropArchivingNodeRecurrent::select_surrogate_gradient( const std::string& surrogate_gradient_function_name )
{
  const auto found_entry_it = surrogate_gradient_funcs_.find( surrogate_gradient_function_name );

  if ( found_entry_it != surrogate_gradient_funcs_.end() )
  {
    return found_entry_it->second;
  }

  std::string error_message = "Surrogate gradient / pseudo-derivate function surrogate_gradient_function from [";
  for ( const auto& surrogate_gradient_func : surrogate_gradient_funcs_ )
  {
    error_message += " \"" + surrogate_gradient_func.first + "\",";
  }
  error_message.pop_back();
  error_message += " ] required.";

  throw BadProperty( error_message );
}


double
EpropArchivingNodeRecurrent::compute_piecewise_linear_surrogate_gradient( const double r,
  const double v_m,
  const double v_th,
  const double beta,
  const double gamma )
{
  if ( r > 0 )
  {
    return 0.0;
  }

  return gamma * std::max( 0.0, 1.0 - beta * std::abs( ( v_m - v_th ) ) );
}

double
EpropArchivingNodeRecurrent::compute_exponential_surrogate_gradient( const double r,
  const double v_m,
  const double v_th,
  const double beta,
  const double gamma )
{
  if ( r > 0 )
  {
    return 0.0;
  }

  return gamma * std::exp( -beta * std::abs( v_m - v_th ) );
}

double
EpropArchivingNodeRecurrent::compute_fast_sigmoid_derivative_surrogate_gradient( const double r,
  const double v_m,
  const double v_th,
  const double beta,
  const double gamma )
{
  if ( r > 0 )
  {
    return 0.0;
  }

  return gamma * std::pow( 1.0 + beta * std::abs( v_m - v_th ), -2 );
}

double
EpropArchivingNodeRecurrent::compute_arctan_surrogate_gradient( const double r,
  const double v_m,
  const double v_th,
  const double beta,
  const double gamma )
{
  if ( r > 0 )
  {
    return 0.0;
  }

  return gamma / M_PI * ( 1.0 / ( 1.0 + std::pow( beta * M_PI * ( v_m - v_th ), 2 ) ) );
}

void
EpropArchivingNodeRecurrent::append_new_eprop_history_entry( const long time_step )
{
  if ( eprop_indegree_ == 0 )
  {
    return;
  }

  eprop_history_.emplace_back( time_step, 0.0, 0.0, 0.0 );
}

void
EpropArchivingNodeRecurrent::write_surrogate_gradient_to_history( const long time_step,
  const double surrogate_gradient )
{
  if ( eprop_indegree_ == 0 )
  {
    return;
  }

  auto it_hist = get_eprop_history( time_step );
  it_hist->surrogate_gradient_ = surrogate_gradient;
}

void
EpropArchivingNodeRecurrent::write_learning_signal_to_history( const long time_step,
  const double learning_signal,
  const bool has_norm_step )
{
  if ( eprop_indegree_ == 0 )
  {
    return;
  }

  long shift = delay_rec_out_ + delay_out_rec_;

  if ( has_norm_step )
  {
    shift += delay_out_norm_;
  }


  auto it_hist = get_eprop_history( time_step - shift );
  const auto it_hist_end = get_eprop_history( time_step - shift + delay_out_rec_ );

  for ( ; it_hist != it_hist_end; ++it_hist )
  {
    it_hist->learning_signal_ += learning_signal;
  }
}

void
EpropArchivingNodeRecurrent::write_firing_rate_reg_to_history( const long t_current_update,
  const double f_target,
  const double c_reg )
{
  if ( eprop_indegree_ == 0 )
  {
    return;
  }

  const double update_interval = kernel().simulation_manager.get_eprop_update_interval().get_steps();
  const double dt = Time::get_resolution().get_ms();
  const long shift = Time::get_resolution().get_steps();

  const double f_av = n_spikes_ / update_interval;
  const double f_target_ = f_target * dt; // convert from spikes/ms to spikes/step
  const double firing_rate_reg = c_reg * ( f_av - f_target_ ) / update_interval;

  firing_rate_reg_history_.emplace_back( t_current_update + shift, firing_rate_reg );
}

void
EpropArchivingNodeRecurrent::write_firing_rate_reg_to_history( const long time_step,
  const long interval_step,
  const double z,
  const double f_target,
  const double kappa_reg,
  const double c_reg )
{
  if ( eprop_indegree_ == 0 )
  {
    return;
  }

  const double dt = Time::get_resolution().get_ms();

  const double f_target_ = f_target * dt; // convert from spikes/ms to spikes/step

  if ( interval_step < 0 )
  {
    return;
  }

  const double kappa_ = interval_step / ( interval_step + 1.0 );
  f_av_ = kappa_ * f_av_ + ( 1.0 - kappa_ ) * z / dt;

  firing_rate_reg_ = c_reg * ( f_av_ - f_target_ );

  auto it_hist = get_eprop_history( time_step );
  it_hist->firing_rate_reg_ = firing_rate_reg_;
}

double
EpropArchivingNodeRecurrent::get_firing_rate_reg_history( const long time_step )
{
  const auto it_hist = std::lower_bound( firing_rate_reg_history_.begin(), firing_rate_reg_history_.end(), time_step );
  assert( it_hist != firing_rate_reg_history_.end() );

  return it_hist->firing_rate_reg_;
}

double
EpropArchivingNodeRecurrent::get_learning_signal_from_history( const long time_step, const bool has_norm_step )
{
  long shift = delay_rec_out_ + delay_out_rec_;

  if ( has_norm_step )
  {
    shift += delay_out_norm_;
  }

  const auto it = get_eprop_history( time_step - shift );
  if ( it == eprop_history_.end() )
  {
    return 0;
  }

  return it->learning_signal_;
}

void
EpropArchivingNodeRecurrent::erase_used_firing_rate_reg_history()
{
  auto it_update_hist = update_history_.begin();
  auto it_reg_hist = firing_rate_reg_history_.begin();

  while ( it_update_hist != update_history_.end() and it_reg_hist != firing_rate_reg_history_.end() )
  {
    if ( it_update_hist->access_counter_ == 0 )
    {
      it_reg_hist = firing_rate_reg_history_.erase( it_reg_hist );
    }
    else
    {
      ++it_reg_hist;
    }
    ++it_update_hist;
  }
}

EpropArchivingNodeReadout::EpropArchivingNodeReadout()
  : EpropArchivingNode()
{
}

EpropArchivingNodeReadout::EpropArchivingNodeReadout( const EpropArchivingNodeReadout& n )
  : EpropArchivingNode( n )
{
}

void
EpropArchivingNodeReadout::append_new_eprop_history_entry( const long time_step, const bool has_norm_step )
{
  if ( eprop_indegree_ == 0 )
  {
    return;
  }

  const long shift = has_norm_step ? delay_out_norm_ : 0;

  eprop_history_.emplace_back( time_step - shift, 0.0 );
}

void
EpropArchivingNodeReadout::write_error_signal_to_history( const long time_step,
  const double error_signal,
  const bool has_norm_step )
{
  if ( eprop_indegree_ == 0 )
  {
    return;
  }

  const long shift = has_norm_step ? delay_out_norm_ : 0;

  auto it_hist = get_eprop_history( time_step - shift );
  it_hist->error_signal_ = error_signal;
}


} // namespace nest
