__doc__ = """ External forces and Actions (in learning) of the snake terrain case."""

import numpy as np
from elastica._linalg import _batch_matvec
from elastica.utils import _bspline

import numba
from numba import njit
from elastica._linalg import (
	_batch_norm,
	_batch_product_i_k_to_ik,
	_batch_product_i_ik_to_k,
	_batch_product_k_ik_to_ik,
	_batch_vec_oneD_vec_cross,
)
from elastica.external_forces import NoForces
from elastica.external_forces import (
	inplace_addition,
	inplace_substraction,
)


class MuscleForces_snake(NoForces):
	"""
	This class applies longitudinal muscle forces along the body. The applied muscle forces are treated
	as external forces applied at each node. This class can apply muscle forces as a traveling wave with a beta spline or only
	as a traveling wave. Application of this class includes mimicking rectilinear locomotion of snake.
	For implementation details refer to Gazzola et. al. RSoS. (2018).

		Attributes
		----------
		angular_frequency: float
			Angular frequency of traveling wave.
		wave_number: float
			Wave number of traveling wave.
		phase_shift: float
			Phase shift of traveling wave.
		ramp_up_time: float
			Applied muscle torques are ramped up until ramp up time.
		my_spline: numpy.ndarray
			1D (blocksize) array containing data with 'float' type. Generated spline.
			
		NEW PARAMETER:
		switch_on_time: float
			time to switch on the muscle activation.
	"""

	def __init__(
		self,
		base_length,
		b_coeff,
		period,
		wave_number,
		phase_shift,
		rest_lengths,
		ramp_up_time=0.0,
		with_spline=False,
		switch_on_time=0.0,
	):
		"""

		Parameters
		----------
		base_length: float
			Rest length of the rod-like object.
		b_coeff: nump.ndarray
			1D array containing data with 'float' type.
			Beta coefficients for beta-spline.
		period: float
			Period of traveling wave.
		wave_number: float
			Wave number of traveling wave.
		phase_shift: float
			Phase shift of traveling wave.
		ramp_up_time: float
			Applied muscle torques are ramped up until ramp up time.
		with_spline: boolean
			Option to use beta-spline.
		
		NEW PARAMETER:
		switch_on_time: float
			time to switch on the muscle activation.
		"""
		super(MuscleForces_snake, self).__init__()

		self.angular_frequency = 2.0 * np.pi / period
		self.wave_number = wave_number
		self.phase_shift = phase_shift

		assert ramp_up_time >= 0.0
		self.ramp_up_time = ramp_up_time
		assert switch_on_time >= 0.0
		self.switch_on_time = switch_on_time

		# s is the position of nodes on the rod, we go from node=0 to node=nelem. For each segment i, we apply
		# forces of same magnitude, opposite directions at node i and node i+1, so that overall internal forces are
		# conserved. Meanwhile, the force magnitude of each segment take value from the beta spline curve defined 
		# as follows:
		
		self.s = np.cumsum(rest_lengths)
		self.s = np.insert(self.s,0,0.0)
		self.s /= self.s[-1]

		if with_spline:
			assert b_coeff.size != 0, "Beta spline coefficient array (t_coeff) is empty"
			my_spline, ctr_pts, ctr_coeffs = _bspline(b_coeff)
			self.my_spline = my_spline(self.s)

		else:

			def constant_function(input):
				"""
				Return array of ones same as the size of the input array. This
				function is called when Beta spline function is not used.

				Parameters
				----------
				input

				Returns
				-------

				"""
				return np.ones(input.shape)

			self.my_spline = constant_function(self.s)

	def apply_forces(self, system, time: np.float64 = 0.0):
		self.compute_muscle_forces(
			time,
			self.my_spline,
			self.s,
			self.angular_frequency,
			self.wave_number,
			self.phase_shift,
			self.ramp_up_time,
			self.switch_on_time,
			system.tangents,
			system.director_collection,
			system.external_forces,
		)

	@staticmethod
	@njit(cache=True)
	def compute_muscle_forces(
		time,
		my_spline,
		s,
		angular_frequency,
		wave_number,
		phase_shift,
		ramp_up_time,
		switch_on_time,
		tangents,
		director_collection,
		external_forces,
	):
		if time > switch_on_time:
			# Ramp up the muscle force
			factor = min(1.0, (time - switch_on_time) / ramp_up_time)
			# From the segment 0 to segment nelem
			# Magnitude of the force. Am(i) = beta((s(i)+s(i+1))/2) * sin(2pi*t/T + 2pi*(s(i)+s(i+1))/2/lambda + phi)
			#s_ = np.array([0.0,s])
			seg_spline = 0.5*(my_spline[:-1]+my_spline[1:])
			seg_s = 0.5*(s[:-1]+s[1:])
			force_mag = (
				factor
				* seg_spline
				* np.sin(angular_frequency * (time - switch_on_time - phase_shift) - wave_number * seg_s)
			)
			# Head and tail of the snake is opposite compared to elastica cpp. We need to iterate torque_mag
			# from last to first element.	
			forces = _batch_product_k_ik_to_ik(force_mag[-1::-1], tangents)
				
			inplace_addition(
				external_forces[..., :-1],
				forces,
			)
			inplace_substraction(
				external_forces[..., 1:],
				forces,
			)

class MuscleTorques_snake(NoForces):
	"""
	This class applies muscle torques along the body. The applied muscle torques are treated
	as applied external forces. This class can apply
	muscle torques as a traveling wave with a beta spline or only
	as a traveling wave. For implementation details refer to Gazzola et. al.
	RSoS. (2018).

		Attributes
		----------
		direction: numpy.ndarray
			2D (dim, 1) array containing data with 'float' type. Muscle torque direction.
		angular_frequency: float
			Angular frequency of traveling wave.
		wave_number: float
			Wave number of traveling wave.
		phase_shift: float
			Phase shift of traveling wave.
		ramp_up_time: float
			Applied muscle torques are ramped up until ramp up time.
		my_spline: numpy.ndarray
			1D (blocksize) array containing data with 'float' type. Generated spline.
			
		NEW PARAMETER:
		switch_on_time: float
			time to switch on the muscle activation.
		is_lateral_wave: bool
			check if it is lateral muscle torque.
	"""

	def __init__(
		self,
		base_length,
		b_coeff,
		period,
		wave_number,
		phase_shift,
		direction,
		rest_lengths,
		ramp_up_time=0.0,
		with_spline=False,
		switch_on_time=0.0,
		is_lateral_wave=True,
	):
		"""

		Parameters
		----------
		base_length: float
			Rest length of the rod-like object.
		b_coeff: nump.ndarray
			1D array containing data with 'float' type.
			Beta coefficients for beta-spline.
		period: float
			Period of traveling wave.
		wave_number: float
			Wave number of traveling wave.
		phase_shift: float
			Phase shift of traveling wave.
		direction: numpy.ndarray
		   1D (dim) array containing data with 'float' type. 
		ramp_up_time: float
			Applied muscle torques are ramped up until ramp up time.
		with_spline: boolean
			Option to use beta-spline.
		
		NEW PARAMETER:
		switch_on_time: float
			time to switch on the muscle activation.
		is_lateral_wave: bool
			check if it is lateral muscle torque.

		"""
		super(MuscleTorques_snake, self).__init__()

		self.direction = direction	# Direction torque applied
		self.angular_frequency = 2.0 * np.pi / period
		self.wave_number = wave_number
		self.phase_shift = phase_shift

		assert ramp_up_time >= 0.0
		self.ramp_up_time = ramp_up_time
		assert switch_on_time >= 0.0
		self.switch_on_time = switch_on_time
		self.is_lateral_wave = is_lateral_wave

		# s is the position of nodes on the rod, we go from node=1 to node=nelem-1, because there is no
		# torques applied by first and last node on elements. Reason is that we cannot apply torque in an
		# infinitesimal segment at the beginning and end of rod, because there is no additional element
		# (at element=-1 or element=n_elem+1) to provide internal torques to cancel out an external
		# torque. This coupled with the requirement that the sum of all muscle torques has
		# to be zero results in this condition.
		self.s = np.cumsum(rest_lengths)
		self.s /= self.s[-1]

		if with_spline:
			assert b_coeff.size != 0, "Beta spline coefficient array (t_coeff) is empty"
			my_spline, ctr_pts, ctr_coeffs = _bspline(b_coeff)
			self.my_spline = my_spline(self.s)

		else:

			def constant_function(input):
				"""
				Return array of ones same as the size of the input array. This
				function is called when Beta spline function is not used.

				Parameters
				----------
				input

				Returns
				-------

				"""
				return np.ones(input.shape)

			self.my_spline = constant_function(self.s)

	def apply_torques(self, system, time: np.float64 = 0.0):
		self.compute_muscle_torques(
			time,
			self.my_spline,
			self.s,
			self.angular_frequency,
			self.wave_number,
			self.phase_shift,
			self.ramp_up_time,
			self.direction,
			self.switch_on_time,
			self.is_lateral_wave,
			system.tangents,
			system.director_collection,
			system.external_torques,
		)

	@staticmethod
	@njit(cache=True)
	def compute_muscle_torques(
		time,
		my_spline,
		s,
		angular_frequency,
		wave_number,
		phase_shift,
		ramp_up_time,
		direction,
		switch_on_time,
		is_lateral_wave,
		tangents,
		director_collection,
		external_torques,
	):
		if time > switch_on_time:
			# Ramp up the muscle torque
			factor = min(1.0, (time - switch_on_time) / ramp_up_time)
			# From the node 1 to node nelem-1
			# Magnitude of the torque. Am = beta(s) * sin(2pi*t/T + 2pi*s/lambda + phi)
			# There is an inconsistency with paper and Elastica cpp implementation. In paper sign in
			# front of wave number is positive, in Elastica cpp it is negative.
			torque_mag = (
				factor
				* my_spline
				* np.sin(angular_frequency * (time - switch_on_time - phase_shift) - wave_number * s)
			)
			# Head and tail of the snake is opposite compared to elastica cpp. We need to iterate torque_mag
			# from last to first element.	
			if (is_lateral_wave):
				torque = _batch_product_i_k_to_ik(direction, torque_mag[-2::-1])
			else:
				# compute torque direction for lifting wave. 
				# Here, direction of each element is computed separately
				# based on the rod tangent and normal direction. This is implemented to
				# correct the binormal direction when snake undergoes lateral bending
				avg_element_direction = 0.5 * (tangents[..., :-1] + tangents[..., 1:])
				torque_direction = _batch_vec_oneD_vec_cross(avg_element_direction, direction)
				torque_direction_unit = _batch_product_k_ik_to_ik(
				1 / (_batch_norm(torque_direction) + 1e-14),
				torque_direction,
				)
				torque = _batch_product_k_ik_to_ik(torque_mag[-2::-1], torque_direction_unit)
				
			inplace_addition(
				external_torques[..., 1:],
				_batch_matvec(director_collection[..., 1:], torque),
			)
			inplace_substraction(
				external_torques[..., :-1],
				_batch_matvec(director_collection[..., :-1], torque),
			)
			
class MuscleTorquesWithRL(NoForces):
	"""
	This class applies lifting muscle torques using Beta spline.
	The muscle torque is characterized by a fixed upper bound amplitude,
	and two paramters - lift_ratio and phase, which are determined through
	learrning process.

	Attributes
	----------
	base_length: float
		Rest length of the rod-like object.
	direction: numpy.ndarray
		2D (dim, 1) array containing data with 'float' type. Normal drection of muscle torque.
	angular_frequency: float
		Angular frequency of traveling wave.
	wave_number: float
		Wave number of traveling wave.
	ramp_up_time: float
		Applied muscle torques are ramped up until ramp up time.
	my_spline: numpy.ndarray
		1D (blocksize) array containing data with 'float' type. Generated spline.
	switch_on_time: float
		time to switch on the muscle activation.
	s: numpy.ndarray
		1D (blocksize) array - unit string coordinate.

	Additional Attributes
	----------
	n_sensor: int
		number of sensor groups
	filter: FPF filter class
		FPF filter
	start: array
		start index of curvature sensing
	end: array
		end index of curvature sensing		  
	"""
	def __init__(
		self,
		base_length,
		b_coeff,
		period,
		wave_number,
		rest_lengths,
		direction,
		step_skip,
		learning_parameters,
		ramp_up_time=0.0,
		with_spline=False,
		switch_on_time=0.0,
		max_rate_of_change_of_activation=0.01,
		**kwargs,
	):
		"""

		Parameters
		----------
		base_length: float
			Rest length of the rod-like object.
		b_coeff: nump.ndarray
			1D array containing data with 'float' type.
			Beta coefficients for beta-spline.
		period: float
			Period of traveling wave.
		wave_number: float
			Wave number of traveling wave.
		rest_lengths: numpy.ndarray
			1D array of segmental lengths
		direction: numpy.ndarray
		   1D (dim) array containing data with 'float' type. Normal drection of muscle torque.
		ramp_up_time: float
			Applied muscle torques are ramped up until ramp up time.
		with_spline: boolean
			Option to use beta-spline.
		switch_on_time: float
			time to switch on the muscle activation.
		learning_parameters: numpy.ndarray
			1D array (two parameters) containing data with 'float' type.
			Stores the values of lift_ratio and phase selected at previous step.
			If controls are changed, learning_parameters_cached updated.
		step_skip : int
			Determines the data collection step.
		max_rate_of_change_of_activation : float
			This limits the maximum change that can happen for control parameters in between two calls of this object.
		**kwargs
			Arbitrary keyword arguments.
		"""
		super(MuscleTorquesWithRL, self).__init__()

		self.base_length = base_length
		self.direction = direction
		self.period = period
		self.angular_frequency = 2.0 * np.pi / period
		self.wave_number = wave_number

		assert ramp_up_time >= 0.0
		self.ramp_up_time = ramp_up_time
		assert switch_on_time >= 0.0
		self.switch_on_time = switch_on_time

		# s is the position of nodes on the rod, we go from node=1 to node=nelem-1, because there is no
		# torques applied by first and last node on elements. Reason is that we cannot apply torque in an
		# infinitesimal segment at the beginning and end of rod, because there is no additional element
		# (at element=-1 or element=n_elem+1) to provide internal torques to cancel out an external
		# torque. This coupled with the requirement that the sum of all muscle torques has
		# to be zero results in this condition.
		self.s = np.cumsum(rest_lengths)
		self.s /= self.s[-1]

		if with_spline:
			assert b_coeff.size != 0, "Beta spline coefficient array (t_coeff) is empty"
			my_spline, ctr_pts, ctr_coeffs = _bspline(b_coeff)
			self.my_spline = my_spline(self.s)

		else:

			def constant_function(input):
				"""
				Return array of ones same as the size of the input array. This
				function is called when Beta spline function is not used.

				Parameters
				----------
				input

				Returns
				-------

				"""
				return np.ones(input.shape)

			self.my_spline = constant_function(self.s)
		
		#RL related parameter initialization.
		self.learning_parameters = (
			learning_parameters
			if hasattr(learning_parameters, "__call__")
			else lambda time_v: learning_parameters
			)
		self.step_skip = step_skip
		self.parameters_profile_recorder = kwargs.get("parameters_profile_recorder", None)	
		self.counter = 0						   #counting output frequency
		self.learning_parameters_cached = np.zeros(2)  # This caches the control parameters. 
		self.max_rate_of_change_of_activation = max_rate_of_change_of_activation
		# Purpose of this flag is to just generate spline even the control points are zero
		# so that code wont crash.
		self.initial_call_flag = 0			  
			

	def apply_torques(self, system, time: np.float64 = 0.0):
		# Check if RL algorithm changed the control parameters
		if (
			not np.array_equal(self.learning_parameters_cached, self.learning_parameters(time))
			or self.initial_call_flag == 0
		):		
			self.initial_call_flag = 1
			
			# Apply filter to the activation signal, to prevent drastic changes in activation signal.
			self.filter_activation(
				self.learning_parameters_cached,
				np.array((self.learning_parameters(time))),
				self.max_rate_of_change_of_activation,
			)			
	
		self.compute_muscle_torques(
			time,
			self.learning_parameters_cached,
			self.my_spline,
			self.s,
			self.period,
			self.angular_frequency,
			self.wave_number,
			self.ramp_up_time,
			self.direction,
			self.switch_on_time,
			system.tangents,
			system.director_collection,
			system.external_torques,
		)
		
		if self.counter % self.step_skip == 0:
			if self.parameters_profile_recorder is not None:
				self.parameters_profile_recorder["time"].append(time)

				self.parameters_profile_recorder["lift_ratio"].append(
					self.learning_parameters_cached[0].copy()
				)
				self.parameters_profile_recorder["phase"].append(
					self.learning_parameters_cached[1].copy()
				)

		self.counter += 1

	@staticmethod
	@njit(cache=True)
	def compute_muscle_torques(
		time,
		learning_parameters_cached,
		my_spline,
		s,
		period,
		angular_frequency,
		wave_number,
		ramp_up_time,
		direction,
		switch_on_time,
		tangents,
		director_collection,
		external_torques,
	):
		if time > switch_on_time:
			# Ramp up the muscle torque
			factor = min(1.0, (time - switch_on_time) / ramp_up_time)
			# unpack learning parameters
			lift_ratio = learning_parameters_cached[0]
			phase_shift = learning_parameters_cached[1] * period
			# From the node 1 to node nelem-1
			# Magnitude of the torque. Am = beta(s) * sin(2pi*t/T + 2pi*s/lambda + phi)
			# There is an inconsistency with paper and Elastica cpp implementation. In paper sign in
			# front of wave number is positive, in Elastica cpp it is negative.
			torque_mag = (
				factor
				* lift_ratio
				* my_spline
				* np.sin(angular_frequency * (time - switch_on_time - phase_shift) - wave_number * s)
			)
			# Head and tail of the snake is opposite compared to elastica cpp. We need to iterate torque_mag
			# from last to first element.	
			# compute torque direction for lateral wave. 
			# Here, direction of each element is computed separately
			# based on the rod tangent and normal direction. This is implemented to
			# correct the binormal direction when snake undergoes lateral bending
			avg_element_direction = 0.5 * (tangents[..., :-1] + tangents[..., 1:])
			torque_direction = _batch_vec_oneD_vec_cross(avg_element_direction, direction)
			torque_direction_unit = _batch_product_k_ik_to_ik(
				1 / (_batch_norm(torque_direction) + 1e-14),
				torque_direction,
			)
			torque = _batch_product_k_ik_to_ik(torque_mag[-2::-1], torque_direction_unit)
				
			inplace_addition(
				external_torques[..., 1:],
				_batch_matvec(director_collection[..., 1:], torque),
			)
			inplace_substraction(
				external_torques[..., :-1],
				_batch_matvec(director_collection[..., :-1], torque),
			)
		
	
	@staticmethod
	@numba.njit(cache=True)
	def filter_activation(signal, input_signal, max_signal_rate_of_change):
		"""
		Filters the input signal. If change in new signal (input signal) greater than
		previous signal (signal) then, increase for signal is max_signal_rate_of_change amount.

		Parameters
		----------
		signal : numpy.ndarray
			1D (number_of_control_points,) array containing data with 'float' type.
		input_signal : numpy.ndarray
			1D (number_of_control_points,) array containing data with 'float' type.
		max_signal_rate_of_change : float
			This limits the maximum change that can happen between signal and input signal.

		Returns
		-------

		"""
		signal_difference = input_signal - signal
		signal += np.sign(signal_difference) * np.minimum(
			max_signal_rate_of_change, np.abs(signal_difference)
		)


class FPFActions(NoForces):
	"""
	This class provides sensory feedback for FPF and applies any corresponding activations

		Attributes
		----------
		n_sensor: int
			number of sensor groups
		filter: FPF filter class
			FPF filter
		start: array
			start index of curvature sensing
		end: array
			end index of curvature sensing		  
			

	"""
	
	def __init__(
		self, 
		n_sensor,
		FPF,
		start_end, 
		dt,
		base_length,
		step_skip,
		):
		"""
		Parameters
		----------
		n_sensor: int
			number of sensor groups
		filter: FPF filter class
			FPF filter
		start: array
			start index of curvature sensing
		end: array
			end index of curvature sensing	 
		"""
		super(FPFActions, self).__init__()

		self.n_sensor = n_sensor
		self.filter = FPF
		self.start = np.array(start_end)[:, 0]
		self.end = np.array(start_end)[:, 1]
		self.dt = dt
		self.base_length = base_length
		self.step_skip = step_skip				   #oscillator steps 
		self.counter = 0						   #counting for oscillator updates
		
	def unit_vector(self, vector):
		return vector / np.linalg.norm(vector)
	
	def angle_between(self, v1, v2):
		v1_u = self.unit_vector(v1)
		v2_u = self.unit_vector(v2)
		return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
	
	def apply_torques(self, system, time: np.float64 = 0.0):
	
		if self.counter % self.step_skip == 0:
			current_curvature = np.zeros(self.n_sensor)
			for i in range(self.n_sensor):
				current_curvature_sign = np.sign(
					np.cross(system.tangents[:,self.start[i]], system.tangents[:,self.end[i]])[2]
				)
				current_curvature[i] = (
					current_curvature_sign
					* self.angle_between(system.tangents[:,self.start[i]], system.tangents[:,self.end[i]])
					/ np.sum(system.rest_lengths[self.start[i]:self.end[i]])
					* self.base_length
				)
			#print(current_curvature)
			y = current_curvature + self.filter.sigma_W / np.sqrt(self.dt) * np.random.randn(self.n_sensor)
			self.filter.time_update(self.dt)
			self.filter.info_update(y, self.dt)
			
			if self.filter.recorder is not None:
				self.filter.recorder["time"].append(time)
				self.filter.recorder["curvature"].append(current_curvature.copy())
		
		self.counter += 1
