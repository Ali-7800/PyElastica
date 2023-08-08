__doc__ = """Snake friction case from X. Zhang et. al. Nat. Comm. 2021"""

import os
import numpy as np
import sys
import argparse
import pickle


sys.path.append("../../")
sys.path.append("../../../../")

from elastica import *

from continuum_snake_postprocessing import (
    plot_snake_velocity,
    plot_video,
    compute_projected_velocity,
    compute_effective_velocity,
    plot_curvature,
)
from snake_ForceAction import (
    MuscleForces_snake,
    MuscleTorques_snake,
    FPFActions,
)
from dissipation_relaxation import *



# oscillators
from FPF import *

def init_FPF(n_sensor, n_particle, omega_0, sigma_W, delta, learning_rate, recorder):
    N = n_particle  
    learning_obsv_flag = True
    # learning_obsv_flag = False
    if learning_obsv_flag:
        sin = np.array([[0.0, 0.0] for ns in range(n_sensor)])
        cos = np.array([[0.0] for ns in range(n_sensor)])
    else:
        learnt_coeff = np.load('learnt_obsv_coefficients.npz')
        sin = learnt_coeff['sin']
        cos = learnt_coeff['cos']
    h_fun = H_fun(cos, sin)
    filter1 = Filter(
        n_particle=N, omega_0=omega_0, sigma_W=sigma_W, delta=delta, 
        h_fun=h_fun, learning=learning_obsv_flag, learning_rate=learning_rate, recorder=recorder)
    return filter1


class SnakeSimulator(BaseSystemCollection, Constraints, Forcing, Damping, CallBacks):
    pass


def run_snake(
    b_coeff_lat, PLOT_FIGURE=False, SAVE_FIGURE=False, SAVE_VIDEO=False, SAVE_RESULTS=False, FPF_OPTION=False, args=[]
):
    # Initialize the simulation class
    snake_sim = SnakeSimulator()
    
    # Get parser args
    phase_space_params = args

    # Simulation parameters
    period = 2.0
    final_time = 10.0
    time_step = phase_space_params.timestep
    total_steps = int(final_time / time_step)
    rendering_fps = 100
    step_skip = int(1.0 / (rendering_fps * time_step))
    
    # collection of snake characteristics 
    n_elem_collect = np.array([25,50])
    base_length_collect = np.array([0.35,0.8])
    base_radius_collect = np.array([0.009,0.009])
    snake_nu_collect = np.array([5e-3,1e-2])
    snake_torque_ratio_collect = np.array([30,20.0])
    
    # select snake to run
    snake_ID = 1
    
    # setting up test params
    n_elem = n_elem_collect[snake_ID]
    base_length = base_length_collect[snake_ID]
    base_radius = base_radius_collect[snake_ID]
    start = np.array([#phase_space_params.start_offset*phase_space_params.block_size, 
                     0.0,
                     -3.2, 
                     -0.16 + base_radius])
    direction = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    density = 1000
    nu = snake_nu_collect[snake_ID]
    E = 1e6
    poisson_ratio = 0.5
    shear_modulus = E / (poisson_ratio + 1.0)

    shearable_rod = CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        0.0,
        E,
        shear_modulus=shear_modulus,
    )

    snake_sim.append(shearable_rod)
    # add damping
    # old damping model (deprecated in v0.3.0) values
    # damping_constant = 2e-3
    # time_step = 8e-6
    damping_constant = phase_space_params.damping
    
    USE_RELAXTION_NU = True
    
    if not USE_RELAXTION_NU:
        # use linear damping with constant damping ratio
        snake_sim.dampen(shearable_rod).using(
            AnalyticalLinearDamper,
            damping_constant=damping_constant,
            time_step=time_step,
        )
    else:
        # use linear damping with exponentially decaying damping ratio
        initial_damping = 10.0 * damping_constant
        relaxation_time = 0.2
        
        snake_sim.dampen(shearable_rod).using(
            AnalyticalLinearDamper_Relaxation,
            damping_constant=damping_constant,
            initial_damping = initial_damping,
            relaxation_time = relaxation_time,
            time_step=time_step,
        )       

    if FPF_OPTION:
        #set up coupled oscillator groups
        n_sensor = 3                 #number of oscillator groups
        n_particle = 100             #number of oscillators in each group
        omega_0 = 2.0*np.pi / period        #angular velocity 
        sigma_W = 0.2               #noise level
        delta = 0.01                 #initial distribution
        learning_rate = 1.0         #update rate for h function (learning radius)
        step_skip_oscillators = int(100)              #number of steps per update
        time_step_oscillators = time_step * step_skip_oscillators
        oscillators_recorder = defaultdict(list)
        
        filter1 = init_FPF(
            n_sensor, n_particle=n_particle, omega_0=omega_0, sigma_W=sigma_W, delta=delta, 
            learning_rate=learning_rate, recorder=oscillators_recorder)     #init
        
        #FPF actions - curvature sensing and feedback activations 
        start_end = [[0, 7], [8, 15], [16, 23]]
        snake_sim.add_forcing_to(shearable_rod).using(
            FPFActions, n_sensor, filter1, start_end, time_step_oscillators, base_length, step_skip_oscillators,
        )

    # Add gravitational forces
    gravitational_acc = -9.80665
    snake_sim.add_forcing_to(shearable_rod).using(
        GravityForces, acc_gravity=np.array([0.0, 0.0, gravitational_acc])
    )

    # 1. Add muscle torques -- lateral wave 
    # Define lateral wave parameters
    lateral_wave_length = phase_space_params.wave_length
    snake_torque_ratio = snake_torque_ratio_collect[snake_ID]
    lateral_amp = b_coeff_lat[:-1]
    
    lateral_ratio = 1.0             # switch of lateral wave
    snake_sim.add_forcing_to(shearable_rod).using(
        MuscleTorques_snake,
        base_length=base_length,
        b_coeff=snake_torque_ratio*lateral_ratio*lateral_amp,
        period=period,
        wave_number=2.0 * np.pi / (lateral_wave_length),
        phase_shift=0.0,
        rest_lengths=shearable_rod.rest_lengths,
        ramp_up_time=period,
        direction=normal,
        with_spline=True,
    )
 
    # 2. Add muscle torques -- lifting wave 
    # Define lifting wave parameters
    lift_wave_length = lateral_wave_length
    lift_amp = np.array(
        #[0.0, 1e-3, 2e-3, 2e-3, 1e-3, 0.0]
       [2e-3, 1.5e-3, 1e-3, 0.0e-3, 0.0e-3, 0.0e-3]
    )
    
    lift_ratio = 0.0                 # switch of lifting wave
    phase = 0.25
    snake_sim.add_forcing_to(shearable_rod).using(
        MuscleTorques_snake,
        base_length=base_length,
        b_coeff=snake_torque_ratio*lift_ratio*lift_amp,
        period=period,
        wave_number=2.0 * np.pi / (lift_wave_length),
        phase_shift=phase*period,
        rest_lengths=shearable_rod.rest_lengths,
        ramp_up_time=0.01,
        direction=normal,
        with_spline=True,
        switch_on_time=4.0,
        is_lateral_wave=False,
    )

    # 3. Add muscle torques -- longitudinal wave 
    # Define longitudinal wave parameters
    longit_wave_length = lateral_wave_length
    longit_amp = np.array(
        #[0.0, 1e-3, 2e-3, 2e-3, 1e-3, 0.0]
       [1, 1, 1, 1, 1, 1]
    )
    
    longit_ratio = 0.0              # switch of longitudinal wave
    phase = 0.0
    snake_sim.add_forcing_to(shearable_rod).using(
        MuscleForces_snake,
        base_length=base_length,
        b_coeff=snake_torque_ratio*longit_ratio*longit_amp,
        period=period,
        wave_number=2.0 * np.pi / (longit_wave_length),
        phase_shift=phase*period,
        rest_lengths=shearable_rod.rest_lengths,
        ramp_up_time=period,
        with_spline=True,
        switch_on_time=0.0,
    )
    
    USE_OBJ_PLANE = True
    # Add frictional plane - Select between two options
    # 1. Traditional friction plane and 3D objects created in Elastica environment:
        # list of available ground and obstacles:
        # None - default, plain ground 
        # "bumpy" - regular bumpy ground (matrix of spheres)
        # "rock" - single sphere representing big rocks
        # "pebble" - matrix of spheres representing region made of pebbles
        # "post" - matrix of rigid posts representing grassy ground region 
    # 2. Complex, realistic 3D terrains defined through external .obj files.
    
    # Some common parameters first - define friction ratio etc.
    slip_velocity_tol = 1e-8
    froude = 0.1    
    mu = base_length / (period * period * np.abs(gravitational_acc) * froude)
    kinetic_mu_array = np.array(
        [mu, 1.5 * mu, 2.0 * mu]
    )  # [forward, backward, sideways]
    static_mu_array = np.zeros(kinetic_mu_array.shape)
    normal_plane = normal
    
    if not USE_OBJ_PLANE:
        # define plane
        origin_plane = np.array([0.0, 0.0, 0.0])
        list_of_grounds = []          #add plane or obstacles here
        write_object = True           #wether or not to write output files of objects for rendering
        
        from CustomFrictionSurface import (
            AnisotropicKineticCoulomb_3DTerrain_Interaction,
        )
        
        # add plane
        snake_sim.add_forcing_to(shearable_rod).using(
            AnisotropicKineticCoulomb_3DTerrain_Interaction,
            k=1e3,
            nu=1e-3,
            plane_origin=origin_plane,
            plane_normal=normal_plane,
            slip_velocity_tol=slip_velocity_tol,
            kinetic_mu_array=kinetic_mu_array,
            list_of_grounds=list_of_grounds,
            write_object=write_object,
            randomize=True,
            block_size = 0.2,
        )
    else:
        # fetch a couple of functions to load and render obj
        from CustomFrictionSurface import (
        CustomInteractionSurfaceGrid,
        CustomFrictionalSurfaceGrid,
        )

        from surface_functions import (
            import_surface_from_obj,
            create_surface_from_parameterization,
            surface_grid,
        )

        povray_viz  = True
        
        # select obj 
        model_path = "3D_models/obj/Mars1"
        max_extent = 5.0
        surface_reorient = [[1, 2],[2, 1]]
        facets,facet_vertex_normals = import_surface_from_obj(
            model_path = model_path,
            max_extent = max_extent,
            max_z = 0.0,
            surface_reorient = surface_reorient,
            povray_viz = povray_viz,
        )

        import_grid = True #set to True if you have a grid already made using surface_import_and_grid.py
        # create grid
        grid_size = max(2*base_radius,base_length/n_elem)
        if not import_grid:
            facets_grid = surface_grid(facets,grid_size)
        else:
            filename = model_path+"/grid_"+str(snake_ID)+"_"+str(max_extent)+".dat"
            with open(filename, "rb") as fptr:
                facets_grid = pickle.load(fptr)

            assert facets_grid["grid_size"] == grid_size, "imported grid has different grid size than for the current snake"
            assert facets_grid["model_path"] == model_path, "imported grid is for a different model"
            assert facets_grid["max_extent"] == max_extent, "imported grid is for a different extent"
            assert facets_grid["surface_reorient"] == [[1, 2],[2, 1]], "imported grid is for a different surface orientation"


        snake_sim.add_forcing_to(shearable_rod).using(
            CustomFrictionalSurfaceGrid,
            k=1e2,
            nu=1e-1,
            facets=facets,
            facet_vertex_normals = facet_vertex_normals,
            facets_grid=facets_grid,
            grid_size=grid_size,
            slip_velocity_tol=slip_velocity_tol,
            static_mu_array=static_mu_array,
            kinetic_mu_array=kinetic_mu_array,
            gamma = 0.1,
        )
    

    # Add call backs
    class ContinuumSnakeCallBack(CallBackBaseClass):
        """
        Call back function for continuum snake
        """

        def __init__(self, step_skip: int, callback_params: dict):
            CallBackBaseClass.__init__(self)
            self.every = step_skip
            self.callback_params = callback_params

        def make_callback(self, system, time, current_step: int):

            if current_step % self.every == 0:

                self.callback_params["time"].append(time)
                self.callback_params["step"].append(current_step)
                self.callback_params["position"].append(
                    system.position_collection.copy()
                )
                self.callback_params["radius"].append(
                    system.radius.copy()
                )
                self.callback_params["velocity"].append(
                    system.velocity_collection.copy()
                )
                self.callback_params["avg_velocity"].append(
                    system.compute_velocity_center_of_mass()
                )

                self.callback_params["center_of_mass"].append(
                    system.compute_position_center_of_mass()
                )
                self.callback_params["curvature"].append(system.kappa.copy())

                return

    pp_list = defaultdict(list)
    snake_sim.collect_diagnostics(shearable_rod).using(
        ContinuumSnakeCallBack, step_skip=step_skip, callback_params=pp_list
    )

    snake_sim.finalize()

    timestepper = PositionVerlet()
    if not USE_OBJ_PLANE:
        integrate(timestepper, snake_sim, final_time, total_steps)
    else:
        from SnakeIntegrator import (
        snake_integrate
        )
        snake_integrate(timestepper, snake_sim, final_time, total_steps) #this integration function stops the integration when snake the leaves the terrain boundary


    if PLOT_FIGURE:
        filename_plot = "continuum_snake_velocity.png"
        plot_snake_velocity(pp_list, period, filename_plot, SAVE_FIGURE)
        plot_curvature(pp_list, shearable_rod.rest_lengths, period, SAVE_FIGURE)

        if SAVE_VIDEO:
            filename_video = "continuum_snake.mp4"
            plot_video(
                pp_list,
                video_name=filename_video,
                fps=rendering_fps,
                xlim=(0, 3),
                ylim=(-1, 1),
            )

    if SAVE_RESULTS:

        filename = "continuum_snake.dat"
        file = open(filename, "wb")
        pickle.dump(pp_list, file)
        file.close()

    # Compute the average forward velocity. These will be used for optimization.
    [_, _, avg_forward, avg_lateral] = compute_projected_velocity(pp_list, period)
    # Compute the effective velocity - how much snake moves per period
    eff_velocity = compute_effective_velocity(pp_list, period)
    # write to file for data collection in phase space running
    file_object = open("outputs.txt","w")
    file_object.writelines("%f"% (eff_velocity))
    file_object.close()
    
    if FPF_OPTION:
        time_stamps = np.array(oscillators_recorder["time"])
        FPF_measurements = np.array(oscillators_recorder["curvature"])
        FPF_h_bar = np.array(oscillators_recorder["h_bar"])
        FPF_theta_phase = np.array(oscillators_recorder["theta_phase"])
        FPF_correction = np.array(oscillators_recorder["correction"])
        print(FPF_h_bar.shape)
        
        num_plot = int(2)

        fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
        plt.rcParams.update({"font.size": 16})
        ax = fig.add_subplot(111)
        ax.grid(b=True, which="minor", color="k", linestyle="--")
        ax.grid(b=True, which="major", color="k", linestyle="-")
        
        for i in range(num_plot):
            ax.plot(
                time_stamps, FPF_measurements[:,i], "r-", label="measurements",
            )
            ax.plot(
                time_stamps, FPF_h_bar[:,i], "k-", label="h_bar",
            )
            ax.plot(
                time_stamps, FPF_theta_phase[:,i], "b-", label="theta_phase",
            )
        
        """
        for i in range(n_particle):
            ax.scatter(
                time_stamps, FPF_correction[:,0,i], s = 5, marker = '+', color = 'C0', alpha = 0.2
            )
        """
        ax.set_xlabel("time [s]", fontsize=16)
        fig.legend(prop={"size": 20})
        plt.show()
        plt.close(plt.gcf())
        
        
        """
        fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
        plt.rcParams.update({"font.size": 16})
        ax = fig.add_subplot(111)
        ax.grid(b=True, which="minor", color="k", linestyle="--")
        ax.grid(b=True, which="major", color="k", linestyle="-")
  
        for i in range(n_particle):
            ax.scatter(
                time_stamps, FPF_omega[...,i], s = 5, marker = '+', color = 'C0', alpha = 0.2
            )
        ax.set_xlabel("time [s]", fontsize=16)
        fig.legend(prop={"size": 20})
        plt.show()
        plt.close(plt.gcf())    
        """ 
    
    

    return avg_forward, avg_lateral, pp_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--wave_length", type=float, default=0.5,
    )
    parser.add_argument(
        "--timestep", type=float, default=5e-4,
    )
    parser.add_argument(
        "--damping", type=float, default=1e-1,
    )
    args = parser.parse_args()
    
    #print(args.wave_length)
    #print(args.block_size)

    # Options
    PLOT_FIGURE = False
    SAVE_FIGURE = False
    SAVE_VIDEO = False
    SAVE_RESULTS = True
    CMA_OPTION = False
    FPF_OPTION = False
    

    if CMA_OPTION:
        import cma

        SAVE_OPTIMIZED_COEFFICIENTS = False

        def optimize_snake(spline_coefficient):
            [avg_forward, _, _] = run_snake(
                spline_coefficient,
                PLOT_FIGURE=False,
                SAVE_FIGURE=False,
                SAVE_VIDEO=False,
                SAVE_RESULTS=False,
            )
            return -avg_forward

        # Optimize snake for forward velocity. In cma.fmin first input is function
        # to be optimized, second input is initial guess for coefficients you are optimizing
        # for and third input is standard deviation you initially set.
        optimized_spline_coefficients = cma.fmin(optimize_snake, 7 * [0], 0.5)

        # Save the optimized coefficients to a file
        filename_data = "optimized_coefficients.txt"
        if SAVE_OPTIMIZED_COEFFICIENTS:
            assert filename_data != "", "provide a file name for coefficients"
            np.savetxt(filename_data, optimized_spline_coefficients, delimiter=",")

    else:
        # Add muscle forces on the rod
        if os.path.exists("optimized_coefficients.txt"):
            t_coeff_optimized = np.genfromtxt(
                "optimized_coefficients.txt", delimiter=","
            )
        else:
            wave_length = 1.0
            t_coeff_optimized = np.array(
                [3.4e-3, 3.3e-3, 5.7e-3, 2.8e-3, 3.0e-3, 3.0e-3]
            )
            t_coeff_optimized = np.hstack((t_coeff_optimized, wave_length))

        # run the simulation
        [avg_forward, avg_lateral, pp_list] = run_snake(
            t_coeff_optimized, PLOT_FIGURE, SAVE_FIGURE, SAVE_VIDEO, SAVE_RESULTS, FPF_OPTION, args
        )

        print("average forward velocity:", avg_forward)
        print("average forward lateral:", avg_lateral)
