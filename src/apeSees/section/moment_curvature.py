"""Neural/teacher-style moment-curvature analysis for fiber sections.

This class can be used for standard moment-curvature analysis or to
generate training data for neural sections by recording detailed
section-level history.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops

from .results import MomentCurvatureResults, SectionResults
from ..timeseries import LinearTimeSeries, ASCE41Protocol

if TYPE_CHECKING:
    from .base import Section
    from ..timeseries import TimeSeries


class MomentCurvature:
    """
    Performs moment-curvature analysis on a fiber section.
    """

    def __init__(self, section: Section):
        self.section: Section = section

    def solve(
        self,
        axial_load: float = 0.0,
        number_of_points: int = 100,
        max_curvature: float = 0.004,
        number_of_iterations: int = 200,
        theta: float = 0.0,
        *,
        use_protocol: bool = False,
        protocol: Optional[TimeSeries] = None,
        record_section_data: bool = False,
    ) -> MomentCurvatureResults:
        """
        Run moment-curvature analysis to a *pre-defined maximum curvature*.
        
        Parameters
        ----------
        axial_load : float, optional
            Applied axial load [N]. Compression is negative. Default is 0.0.
        number_of_points : int, optional
            Number of analysis steps. Default is 100.
        max_curvature : float, optional
            Maximum target curvature [1/mm]. Default is 0.004.
        number_of_iterations : int, optional
            Maximum iterations per step. Default is 200.
        theta : float, optional
            Section rotation angle [degrees]. Default is 0.0.
        use_protocol : bool, optional
            If True and protocol is provided, use cyclic protocol. Default is False.
        protocol : TimeSeries, optional
            Custom loading protocol. If None and use_protocol=True, uses ASCE41.
        record_section_data : bool, optional
            If True, records section deformation, force, and tangent data. Default is False.
        
        Returns
        -------
        MomentCurvatureResults
            Analysis results containing curvature, moment, and fiber history.
        """
        
        # --- OpenSees model setup ---
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)

        BC_node = 1
        load_node = 2
        ops.node(BC_node, 0.0, 0.0, 0.0)
        ops.node(load_node, 0.0, 0.0, 0.0)

        ops.fix(BC_node, *[1, 1, 1, 1, 1, 1])
        ops.fix(load_node, *[0, 1, 1, 1, 0, 1]) 

        # --- THIS IS THE FIX ---
        # The section's .build() method (from GeneralFiberSection)
        # is now responsible for building its own materials.
        section_tag = self.section.build()
        # --- END FIX ---

        element_tag = 200
        theta_rad = np.radians(theta)
        ops.element(
            'zeroLengthSection', element_tag,
            BC_node, load_node, section_tag,
            '-orient', 1.0, 0.0, 0.0,
                     0.0, -np.cos(theta_rad), np.sin(theta_rad)
        )

        # --- Axial load ---
        ts_ax = 500
        ops.timeSeries('Constant', ts_ax)
        pat_ax = 600
        ops.pattern('Plain', pat_ax, ts_ax)
        ops.load(load_node, axial_load, 0, 0, 0, 0, 0)

        ops.integrator('LoadControl', 0.0)
        ops.system('SuperLU')
        ops.test('NormUnbalance', 1e-6, number_of_iterations, 0)
        ops.numberer('Plain')
        ops.constraints('Transformation')
        ops.algorithm('KrylovNewton')
        ops.analysis('Static')
        ops.analyze(1)

        # --- Time series for MC ---
        timeseries_tag = 501
        if use_protocol and protocol is not None:
            protocol.build()
            timeseries_tag = protocol.tag
        elif use_protocol:
            ASCE41Protocol(tag=timeseries_tag, max_disp=1.0).build()
        else:
            LinearTimeSeries(timeseries_tag, factor=1.0).build()

        # --- Pattern for MC ---
        pat_mc = 601
        ops.pattern('Plain', pat_mc, timeseries_tag)
        ops.sp(load_node, 5, max_curvature)

        dκ = 1.0 / number_of_points
        ops.integrator('LoadControl', dκ)

        # --- Allocate results ---
        n_steps = number_of_points + 1
        curvatures = np.zeros(n_steps)
        moments = np.zeros(n_steps)
        
        section_results = None
        if record_section_data:
            section_deformation = np.zeros((n_steps, 4))
            section_force = np.zeros((n_steps, 4))
            section_tangent = np.zeros((n_steps, 4, 4))
            section_results = SectionResults(
                deformation=section_deformation,
                force=section_force,
                tangent=section_tangent,
            )
            
        probe = ops.eleResponse(element_tag, 'section', 'fiberData2')
        n_fib = len(probe) // 6
        fiber_history = np.zeros((n_steps, n_fib, 6))
        fiber_history[0, :, :] = np.array(probe, dtype=float).reshape(-1, 6)

        # --- Run analysis ---
        converged = True
        stop_reason = "Analysis completed to max_curvature" # Default stop reason
        
        for i in range(n_steps):
            curvatures[i] = ops.nodeDisp(load_node, 5)
            moments[i] = ops.eleForce(element_tag, 5)

            if record_section_data:
                sec_def = ops.eleResponse(element_tag, 'section', 'deformation')
                sec_force = ops.eleResponse(element_tag, 'section', 'force')
                sec_kt = ops.eleResponse(element_tag, 'section', 'stiffness')
                section_deformation[i, :] = np.array(sec_def, dtype=float)
                section_force[i, :] = np.array(sec_force, dtype=float)
                section_tangent[i, :, :] = np.array(sec_kt, dtype=float).reshape(4, 4)

            fdat = np.array(
                ops.eleResponse(element_tag, 'section', 'fiberData2'),
                dtype=float
            )
            fiber_history[i, :, :] = fdat.reshape(-1, 6)

            if i < n_steps - 1:
                if ops.analyze(1) != 0:
                    converged = False
                    stop_reason = f"Analysis failed to converge at step {i+1}"
                    # Truncate all arrays consistently
                    curvatures = curvatures[:i+1]
                    moments = moments[:i+1]
                    fiber_history = fiber_history[:i+1, :, :]
                    if record_section_data:
                        section_deformation = section_deformation[:i+1, :]
                        section_force = section_force[:i+1, :]
                        section_tangent = section_tangent[:i+1, :, :]
                    break
        ops.wipe()
        
        return MomentCurvatureResults(
            axial_load=axial_load,
            curvatures=curvatures,
            moments=moments,
            fiber_history=fiber_history,
            section=section_results,
            theta=theta,
            max_curvature=max_curvature,
            converged=converged,
            stop_reason=stop_reason
        )

    # --- NEW METHOD ---
    def solve_by_strain(
        self,
        axial_load: float = 0.0,
        curvature_increment: float = 1e-5,
        max_steps: int = 1000,
        number_of_iterations: int = 200,
        theta: float = 0.0,
        record_section_data: bool = False,
    ) -> MomentCurvatureResults:
        """
        Run moment-curvature analysis, stopping when any material
        exceeds its defined strain limit.
        """
        
        # --- 1. Get strain limits from the section's materials ---
        all_materials = self.section.get_materials()
        if not all_materials:
            raise ValueError("Section has no materials to get strain limits from.")
            
        # Find the "tightest" bounds from all materials
        max_tensile_strain = min(m.max_tensile_strain for m in all_materials)
        max_compressive_strain = max(m.max_compressive_strain for m in all_materials)
        
        print(f"Running strain-controlled analysis with limits: "
              f"[{max_compressive_strain:.2e}, {max_tensile_strain:.2e}]")
        
        # --- 2. Setup (Same as solve) ---
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)
        BC_node, load_node = 1, 2
        ops.node(BC_node, 0.0, 0.0, 0.0)
        ops.node(load_node, 0.0, 0.0, 0.0)
        ops.fix(BC_node, *[1, 1, 1, 1, 1, 1])
        ops.fix(load_node, *[0, 1, 1, 1, 0, 1])
        section_tag = self.section.build(verbose=False) # Build section
        element_tag = 200
        theta_rad = np.radians(theta)
        ops.element(
            'zeroLengthSection', element_tag,
            BC_node, load_node, section_tag,
            '-orient', 1.0, 0.0, 0.0,
                     0.0, -np.cos(theta_rad), np.sin(theta_rad)
        )
        
        # --- 3. Axial Load (Same as solve) ---
        ts_ax = 500
        ops.timeSeries('Constant', ts_ax)
        pat_ax = 600
        ops.pattern('Plain', pat_ax, ts_ax)
        ops.load(load_node, axial_load, 0, 0, 0, 0, 0)
        ops.integrator('LoadControl', 0.0)
        ops.system('SuperLU')
        ops.test('NormUnbalance', 1e-6, number_of_iterations, 0)
        ops.numberer('Plain')
        ops.constraints('Transformation')
        ops.algorithm('KrylovNewton')
        ops.analysis('Static')
        ops.analyze(1)

        # --- 4. Moment-Curvature Pattern (KEY CHANGE) ---
        timeseries_tag = 501
        LinearTimeSeries(timeseries_tag, factor=1.0).build() # Monotonic only
        pat_mc = 601
        ops.pattern('Plain', pat_mc, timeseries_tag)
        ops.sp(load_node, 5, 1.0) 
        ops.integrator('LoadControl', curvature_increment) 
        
        # --- 5. Data Storage (use lists, size is unknown) ---
        curvatures_list: List[float] = [ops.nodeDisp(load_node, 5)]
        moments_list: List[float] = [ops.eleForce(element_tag, 5)]
        
        probe = ops.eleResponse(element_tag, 'section', 'fiberData2')
        fiber_history_list: List[np.ndarray] = [
            np.array(probe, dtype=float).reshape(-1, 6)
        ]
        
        sec_def_list, sec_force_list, sec_tangent_list = [], [], []
        if record_section_data:
            sec_def_list.append(ops.eleResponse(element_tag, 'section', 'deformation'))
            sec_force_list.append(ops.eleResponse(element_tag, 'section', 'force'))
            sec_tangent_list.append(np.array(
                ops.eleResponse(element_tag, 'section', 'stiffness'), 
                dtype=float
            ).reshape(4, 4))

        # --- 6. Strain-Controlled Analysis Loop (KEY CHANGE) ---
        converged = True
        stop_reason = None  # <-- Initialize stop_reason
        
        for i in range(max_steps):
            if ops.analyze(1) != 0:
                converged = False
                stop_reason = f"Analysis failed to converge at step {i+1}"
                print(stop_reason)
                break
            
            # --- 7. Record Data ---
            curvatures_list.append(ops.nodeDisp(load_node, 5))
            moments_list.append(ops.eleForce(element_tag, 5))
            
            fdat = np.array(
                ops.eleResponse(element_tag, 'section', 'fiberData2'), 
                dtype=float
            ).reshape(-1, 6)
            fiber_history_list.append(fdat)
            
            if record_section_data:
                sec_def_list.append(ops.eleResponse(element_tag, 'section', 'deformation'))
                sec_force_list.append(ops.eleResponse(element_tag, 'section', 'force'))
                sec_tangent_list.append(np.array(
                    ops.eleResponse(element_tag, 'section', 'stiffness'), 
                    dtype=float
                ).reshape(4, 4))
            
            # --- 8. Check Strain Limits ---
            strains = fdat[:, 5]  # Strain is the 6th column (index 5)
            current_max_comp = np.min(strains)
            current_max_tens = np.max(strains)
            
            if current_max_comp <= max_compressive_strain:
                stop_reason = f"Compressive limit reached ({current_max_comp:.2e} <= {max_compressive_strain:.2e})"
            elif current_max_tens >= max_tensile_strain:
                stop_reason = f"Tensile limit reached ({current_max_tens:.2e} >= {max_tensile_strain:.2e})"
            
            if stop_reason:
                print(f"Analysis stopped at step {i+1}: {stop_reason}")
                break
        
        if i == max_steps - 1 and not stop_reason:
            stop_reason = f"Max steps ({max_steps}) reached"
            print(f"Warning: {stop_reason} before target strain.")
            
        ops.wipe()

        # --- 9. Post-Process (Convert lists to arrays) ---
        curvatures_arr = np.array(curvatures_list)
        moments_arr = np.array(moments_list)
        fiber_history_arr = np.array(fiber_history_list)
        
        section_results = None
        if record_section_data:
            section_results = SectionResults(
                deformation=np.array(sec_def_list, dtype=float),
                force=np.array(sec_force_list, dtype=float),
                tangent=np.array(sec_tangent_list, dtype=float)
            )

        return MomentCurvatureResults(
            axial_load=axial_load,
            curvatures=curvatures_arr,
            moments=moments_arr,
            fiber_history=fiber_history_arr,
            section=section_results,
            theta=theta,
            max_curvature=curvatures_arr[-1], # The max curvature we reached
            converged=converged,
            stop_reason=stop_reason  # <-- NEW: Save the reason
        )

    def plot(
        self,
        axial_load: float = 0.0,
        number_of_points: int = 100,
        max_curvature: float = 0.004,
        number_of_iterations: int = 200,
        theta: float = 0.0,
        use_protocol: bool = False,
        protocol: Optional[TimeSeries] = None,
        ax: Optional[plt.Axes] = None,
        label: Optional[str] = None,
        **plot_kwargs,
    ) -> Tuple[plt.Axes, MomentCurvatureResults]:
        
        result = self.solve(
            axial_load=axial_load,
            number_of_points=number_of_points,
            max_curvature=max_curvature,
            number_of_iterations=number_of_iterations,
            theta=theta,
            use_protocol=use_protocol,
            protocol=protocol,
        )
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))
        if label is None:
            label = f"P = {result.axial_load:.2e} N, θ = {result.theta}°"
        ax.plot(result.curvatures, result.moments, label=label, **plot_kwargs)
        ax.set_xlabel("Curvature")
        ax.set_ylabel("Moment")
        ax.set_title("Moment-Curvature Response")
        ax.grid(True, alpha=0.3)
        ax.legend()
        return ax, result

    # --- NEW PLOT HELPER ---
    def plot_by_strain(
        self,
        axial_load: float = 0.0,
        curvature_increment: float = 1e-5,
        max_steps: int = 1000,
        number_of_iterations: int = 200,
        theta: float = 0.0,
        ax: Optional[plt.Axes] = None,
        label: Optional[str] = None,
        **plot_kwargs,
    ) -> Tuple[plt.Axes, MomentCurvatureResults]:
        """
        Runs `solve_by_strain` and plots the result.
        """
        result = self.solve_by_strain(
            axial_load=axial_load,
            curvature_increment=curvature_increment,
            max_steps=max_steps,
            number_of_iterations=number_of_iterations,
            theta=theta,
        )

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        if label is None:
            label = f"P = {result.axial_load:.2e} N, θ = {result.theta}°"

        ax.plot(result.curvatures, result.moments, label=label, **plot_kwargs)
        
        ax.set_xlabel("Curvature")
        ax.set_ylabel("Moment")
        ax.set_title("Moment-Curvature Response (Strain-Controlled)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        return ax, result