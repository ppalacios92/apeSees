"""Neural/teacher-style moment-curvature analysis for fiber sections.

This class implements a Composite Pattern to separate solving logic.
- MomentCurvature: The main class, holds the section.
- _MomentCurvatureSolvers: An internal class attached as 'solve',
  which contains the 'monotonic_by_curvature', 'monotonic_by_strain', 
  and 'cyclic' analysis methods.
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


# --- Solver Class with Explicit Methods ---
class _MomentCurvatureSolvers:
    """
    Internal class holding the solving logic.
    This is accessed via `MomentCurvature.solve`
    """
    def __init__(self, section: Section):
        self.section: Section = section

    # --- METHOD 1: BY CURVATURE ---
    def monotonic_by_curvature(
        self,
        axial_load: float = 0.0,
        *,
        max_curvature: float = 0.001,
        number_of_points: int = 1000,
        number_of_iterations: int = 400,
        theta: float = 0.0,
        record_section_data: bool = False,
    ) -> MomentCurvatureResults:
        """
        Runs a monotonic analysis, stopping at a target curvature.
        
        This analysis does NOT check for material strain limits.
        It will run for exactly `number_of_points`.
        """
        print(f"--- Running Monotonic Analysis to Target Curvature: {max_curvature:.2e} ---")
        
        # --- 2. Setup & 3. Axial Load ---
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)
        BC_node, load_node = 1, 2
        ops.node(BC_node, 0.0, 0.0, 0.0)
        ops.node(load_node, 0.0, 0.0, 0.0)
        ops.fix(BC_node, *[1, 1, 1, 1, 1, 1])
        ops.fix(load_node, *[0, 1, 1, 1, 0, 1])
        section_tag = self.section.build(verbose=False)
        element_tag = 200
        theta_rad = np.radians(theta)
        ops.element('zeroLengthSection', element_tag, BC_node, load_node, section_tag,
                    '-orient', 1.0, 0.0, 0.0, 0.0, -np.cos(theta_rad), np.sin(theta_rad))
        
        ops.timeSeries('Constant', 500)
        ops.pattern('Plain', 600, 500)
        ops.load(load_node, axial_load, 0, 0, 0, 0, 0)
        ops.integrator('LoadControl', 0.0)
        ops.system('SuperLU')
        ops.test('NormUnbalance', 1e-6, number_of_iterations, 0)
        ops.numberer('Plain')
        ops.constraints('Transformation')
        ops.algorithm('KrylovNewton')
        ops.analysis('Static')
        ops.analyze(1)

        # --- 4. Monotonic Pattern Setup (by_curvature logic) ---
        ops.timeSeries('Linear', 501, '-factor', 1.0)
        ops.pattern('Plain', 601, 501)
        ops.sp(load_node, 5, max_curvature) # Apply target curvature
        d_inc = 1.0 / number_of_points     # Increment is a fraction
        ops.integrator('LoadControl', d_inc)
        
        # --- 5. Data Storage ---
        curvatures_list = [ops.nodeDisp(load_node, 5)]
        moments_list = [ops.eleForce(element_tag, 5)]
        probe = ops.eleResponse(element_tag, 'section', 'fiberData2')
        fiber_history_list = [np.array(probe, dtype=float).reshape(-1, 6)]
        sec_def_list, sec_force_list, sec_tangent_list = [], [], []
        if record_section_data:
            sec_def_list.append(ops.eleResponse(element_tag, 'section', 'deformation'))
            sec_force_list.append(ops.eleResponse(element_tag, 'section', 'force'))
            sec_tangent_list.append(np.array(ops.eleResponse(element_tag, 'section', 'stiffness'), dtype=float).reshape(4, 4))

        # --- 6. Analysis Loop ---
        converged = True
        stop_reason = None
        
        for i in range(number_of_points):
            if ops.analyze(1) != 0:
                converged = False
                stop_reason = f"Analysis failed to converge at step {i+1}"
                print(stop_reason)
                break
            
            # Record Data
            curvatures_list.append(ops.nodeDisp(load_node, 5))
            moments_list.append(ops.eleForce(element_tag, 5))
            fdat = np.array(ops.eleResponse(element_tag, 'section', 'fiberData2'), dtype=float).reshape(-1, 6)
            fiber_history_list.append(fdat)
            if record_section_data:
                sec_def_list.append(ops.eleResponse(element_tag, 'section', 'deformation'))
                sec_force_list.append(ops.eleResponse(element_tag, 'section', 'force'))
                sec_tangent_list.append(np.array(ops.eleResponse(element_tag, 'section', 'stiffness'), dtype=float).reshape(4, 4))
            
            # NO STRAIN CHECK
        
        if i == number_of_points - 1 and not stop_reason:
            stop_reason = f"Analysis completed {number_of_points} steps to target curvature {max_curvature:.2e}"
                     
        ops.wipe()

        # --- 10. Post-Process ---
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
            curvatures=curvatures_arr, moments=moments_arr,
            fiber_history=fiber_history_arr, section=section_results,
            theta=theta, converged=converged, stop_reason=stop_reason
        )

    # --- METHOD 2: BY STRAIN ---
    def monotonic_by_strain(
        self,
        axial_load: float = 0.0,
        *,
        curvature_increment: float = 1e-6,
        max_steps: int = 2000,
        number_of_iterations: int = 400,
        theta: float = 0.0,
        record_section_data: bool = False,
        _verbose: bool = True # Internal flag
    ) -> MomentCurvatureResults:
        """
        Runs a monotonic analysis, stopping at the first material strain limit.
        
        This analysis has NO curvature target.
        """
        if _verbose: print("--- Running Monotonic Analysis to Strain Limit ---")
        
        # --- 1. Get strain limits ---
        all_materials = self.section.get_materials()
        if not all_materials:
            raise ValueError("Section has no materials to get strain limits from.")
        max_tensile_strain = min(m.max_tensile_strain for m in all_materials)
        max_compressive_strain = max(m.max_compressive_strain for m in all_materials)
        if _verbose:
            print(f"Strain limits: [{max_compressive_strain:.2e}, {max_tensile_strain:.2e}]")
        
        # --- 2. Setup & 3. Axial Load ---
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)
        BC_node, load_node = 1, 2
        ops.node(BC_node, 0.0, 0.0, 0.0)
        ops.node(load_node, 0.0, 0.0, 0.0)
        ops.fix(BC_node, *[1, 1, 1, 1, 1, 1])
        ops.fix(load_node, *[0, 1, 1, 1, 0, 1])
        section_tag = self.section.build(verbose=False)
        element_tag = 200
        theta_rad = np.radians(theta)
        ops.element('zeroLengthSection', element_tag, BC_node, load_node, section_tag,
                    '-orient', 1.0, 0.0, 0.0, 0.0, -np.cos(theta_rad), np.sin(theta_rad))
        
        ops.timeSeries('Constant', 500)
        ops.pattern('Plain', 600, 500)
        ops.load(load_node, axial_load, 0, 0, 0, 0, 0)
        ops.integrator('LoadControl', 0.0)
        ops.system('SuperLU')
        ops.test('NormUnbalance', 1e-6, number_of_iterations, 0)
        ops.numberer('Plain')
        ops.constraints('Transformation')
        ops.algorithm('KrylovNewton')
        ops.analysis('Static')
        ops.analyze(1)

        # --- 4. Monotonic Pattern Setup (by_strain logic) ---
        ops.timeSeries('Linear', 501, '-factor', 1.0)
        ops.pattern('Plain', 601, 501)
        ops.sp(load_node, 5, 1.0) # Load is 1.0
        ops.integrator('LoadControl', curvature_increment) # Increment is step size
        
        # --- 5. Data Storage ---
        curvatures_list = [ops.nodeDisp(load_node, 5)]
        moments_list = [ops.eleForce(element_tag, 5)]
        probe = ops.eleResponse(element_tag, 'section', 'fiberData2')
        fiber_history_list = [np.array(probe, dtype=float).reshape(-1, 6)]
        sec_def_list, sec_force_list, sec_tangent_list = [], [], []
        if record_section_data:
            sec_def_list.append(ops.eleResponse(element_tag, 'section', 'deformation'))
            sec_force_list.append(ops.eleResponse(element_tag, 'section', 'force'))
            sec_tangent_list.append(np.array(ops.eleResponse(element_tag, 'section', 'stiffness'), dtype=float).reshape(4, 4))

        # --- 6. Analysis Loop ---
        converged = True
        stop_reason = None
        
        for i in range(max_steps):
            if ops.analyze(1) != 0:
                converged = False
                stop_reason = f"Analysis failed to converge at step {i+1}"
                if _verbose: print(stop_reason)
                break
            
            # Record Data
            curvatures_list.append(ops.nodeDisp(load_node, 5))
            moments_list.append(ops.eleForce(element_tag, 5))
            fdat = np.array(ops.eleResponse(element_tag, 'section', 'fiberData2'), dtype=float).reshape(-1, 6)
            fiber_history_list.append(fdat)
            if record_section_data:
                sec_def_list.append(ops.eleResponse(element_tag, 'section', 'deformation'))
                sec_force_list.append(ops.eleResponse(element_tag, 'section', 'force'))
                sec_tangent_list.append(np.array(ops.eleResponse(element_tag, 'section', 'stiffness'), dtype=float).reshape(4, 4))
            
            # Check Stopping Conditions (Strain ONLY)
            strains = fdat[:, 5]
            current_max_comp = np.min(strains)
            current_max_tens = np.max(strains)
            
            if current_max_comp <= max_compressive_strain:
                stop_reason = f"Compressive limit reached ({current_max_comp:.2e} <= {max_compressive_strain:.2e})"
            elif current_max_tens >= max_tensile_strain:
                stop_reason = f"Tensile limit reached ({current_max_tens:.2e} >= {max_tensile_strain:.2e})"
            
            if stop_reason:
                if _verbose: print(f"Analysis stopped at step {i+1}: {stop_reason}")
                break
        
        if i == max_steps - 1 and not stop_reason:
            stop_reason = f"Max steps ({max_steps}) reached before strain limit"
            if _verbose: print(f"Warning: {stop_reason}.")
                     
        ops.wipe()

        # --- 10. Post-Process ---
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
            curvatures=curvatures_arr, moments=moments_arr,
            fiber_history=fiber_history_arr, section=section_results,
            theta=theta, converged=converged, stop_reason=stop_reason
        )

    # --- METHOD 3: CYCLIC ---
    def cyclic(
        self,
        axial_load: float = 0.0,
        *,
        protocol: Optional[TimeSeries] = None,
        number_of_points: int = 1000,
        curvature_scale_factor: float = None,
        scale_protocol_to_strain_limit: bool = True,
        number_of_iterations: int = 400,
        theta: float = 0.0,
        record_section_data: bool = False,
        check_strain_limits: bool = True,
        _verbose: bool = True,
    ) -> MomentCurvatureResults:
        """
        Runs a protocol-based cyclic analysis.
        
        By default (`scale_protocol_to_strain_limit=True`), it will *first* run
        `self.monotonic_by_strain()` to find the strain-limited curvature, and
        use *that* as the scale factor.
        
        If `scale_protocol_to_strain_limit=False`, you *must* provide
        a `curvature_scale_factor`.
        """
        
        if scale_protocol_to_strain_limit is False and curvature_scale_factor is None:
            raise ValueError(
                "If 'scale_protocol_to_strain_limit' is False, "
                "'curvature_scale_factor' must be provided."
            )
        
        if scale_protocol_to_strain_limit:
            if _verbose:
                print("--- Running Smart Cyclic Analysis ---")
                print("Step 1: Running strain-limited monotonic analysis to find scale factor...")
            
            # Use the new, dedicated "by_strain" method
            mono_results = self.monotonic_by_strain(
                axial_load=axial_load,
                theta=theta,
                number_of_iterations=number_of_iterations,
                _verbose=False, 
            )
            
            if not mono_results.converged or "limit" not in (mono_results.stop_reason or ""):
                raise RuntimeError(
                    "Monotonic pre-run failed to find a strain limit. "
                    f"Stop reason: {mono_results.stop_reason}"
                )
            
            final_scale_factor = mono_results.peak_curvature
            if _verbose:
                print(f"Step 1 Complete: Found strain-limited curvature = {final_scale_factor:.3e}")
                print("Step 2: Running cyclic analysis scaled to this curvature...")
        else:
            final_scale_factor = curvature_scale_factor
            if _verbose:
                print("--- Running Cyclic Analysis ---")
                print(f"Running cyclic analysis with manual scale factor = {final_scale_factor:.3e}")
        
        # --- 1. Get strain limits ---
        if check_strain_limits:
            all_materials = self.section.get_materials()
            if not all_materials:
                raise ValueError("Section has no materials to get strain limits from.")
            max_tensile_strain = min(m.max_tensile_strain for m in all_materials)
            max_compressive_strain = max(m.max_compressive_strain for m in all_materials)
            if _verbose:
                print(f"Strain limits: [{max_compressive_strain:.2e}, {max_tensile_strain:.2e}]")
        else:
            max_tensile_strain, max_compressive_strain = np.inf, -np.inf
            if _verbose:
                print("Running protocol. Strain limits will not be checked.")
        
        # --- 2. Setup & 3. Axial Load ---
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)
        BC_node, load_node = 1, 2
        ops.node(BC_node, 0.0, 0.0, 0.0)
        ops.node(load_node, 0.0, 0.0, 0.0)
        ops.fix(BC_node, *[1, 1, 1, 1, 1, 1])
        ops.fix(load_node, *[0, 1, 1, 1, 0, 1])
        section_tag = self.section.build(verbose=False)
        element_tag = 200
        theta_rad = np.radians(theta)
        ops.element('zeroLengthSection', element_tag, BC_node, load_node, section_tag,
                    '-orient', 1.0, 0.0, 0.0, 0.0, -np.cos(theta_rad), np.sin(theta_rad))
        
        ops.timeSeries('Constant', 500)
        ops.pattern('Plain', 600, 500)
        ops.load(load_node, axial_load, 0, 0, 0, 0, 0)
        ops.integrator('LoadControl', 0.0)
        ops.system('SuperLU')
        ops.test('NormUnbalance', 1e-6, number_of_iterations, 0)
        ops.numberer('Plain')
        ops.constraints('Transformation')
        ops.algorithm('KrylovNewton')
        ops.analysis('Static')
        ops.analyze(1)

        # --- 4. Cyclic Pattern Setup ---
        timeseries_tag = 501
        if protocol is not None:
            protocol.build()
            timeseries_tag = protocol.tag
        else:
            if _verbose: print("Using default ASCE41Protocol.")
            ASCE41Protocol(tag=timeseries_tag, max_disp=1.0).build()
        
        ops.pattern('Plain', 601, timeseries_tag)
        ops.sp(load_node, 5, final_scale_factor) 
        d_inc = 1.0 / number_of_points
        ops.integrator('LoadControl', d_inc)
        
        # --- 5. Data Storage ---
        curvatures_list = [ops.nodeDisp(load_node, 5)]
        moments_list = [ops.eleForce(element_tag, 5)]
        probe = ops.eleResponse(element_tag, 'section', 'fiberData2')
        fiber_history_list = [np.array(probe, dtype=float).reshape(-1, 6)]
        sec_def_list, sec_force_list, sec_tangent_list = [], [], []
        if record_section_data:
            sec_def_list.append(ops.eleResponse(element_tag, 'section', 'deformation'))
            sec_force_list.append(ops.eleResponse(element_tag, 'section', 'force'))
            sec_tangent_list.append(np.array(ops.eleResponse(element_tag, 'section', 'stiffness'), dtype=float).reshape(4, 4))

        # --- 6. Analysis Loop ---
        converged = True
        stop_reason = None
        
        for i in range(number_of_points):
            if ops.analyze(1) != 0:
                converged = False
                stop_reason = f"Analysis failed to converge at step {i+1}"
                if _verbose: print(stop_reason)
                break
            
            # Record Data
            curvatures_list.append(ops.nodeDisp(load_node, 5))
            moments_list.append(ops.eleForce(element_tag, 5))
            fdat = np.array(ops.eleResponse(element_tag, 'section', 'fiberData2'), dtype=float).reshape(-1, 6)
            fiber_history_list.append(fdat)
            if record_section_data:
                sec_def_list.append(ops.eleResponse(element_tag, 'section', 'deformation'))
                sec_force_list.append(ops.eleResponse(element_tag, 'section', 'force'))
                sec_tangent_list.append(np.array(ops.eleResponse(element_tag, 'section', 'stiffness'), dtype=float).reshape(4, 4))
            
            # Check Stopping Conditions
            strains = fdat[:, 5]
            current_max_comp = np.min(strains)
            current_max_tens = np.max(strains)
            
            if current_max_comp <= max_compressive_strain:
                stop_reason = f"Compressive limit reached ({current_max_comp:.2e} <= {max_compressive_strain:.2e})"
            elif current_max_tens >= max_tensile_strain:
                stop_reason = f"Tensile limit reached ({current_max_tens:.2e} >= {max_tensile_strain:.2e})"
            
            if stop_reason:
                if _verbose: print(f"Analysis stopped at step {i+1}: {stop_reason}")
                break
        
        if i == number_of_points - 1 and not stop_reason:
            stop_reason = f"Analysis completed protocol ({number_of_points} steps)"
                     
        ops.wipe()

        # --- 10. Post-Process ---
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
            curvatures=curvatures_arr, moments=moments_arr,
            fiber_history=fiber_history_arr, section=section_results,
            theta=theta, converged=converged, stop_reason=stop_reason
        )

# --- Main MomentCurvature Class (Refactored) ---

class MomentCurvature:
    """
    Performs moment-curvature analysis on a fiber section.
    
    This class holds the section and provides solver methods via the
    `.solve` attribute (e.S., `mc.solve.monotonic_by_strain(...)`).
    """

    def __init__(self, section: Section):
        self.section: Section = section
        # --- COMPOSITE PATTERN ---
        self.solve: _MomentCurvatureSolvers = _MomentCurvatureSolvers(self.section)

    # --- New Plot Helpers ---
    
    def plot_monotonic_by_curvature(
        self,
        axial_load: float = 0.0,
        *,
        max_curvature: float = 0.01,
        number_of_points: int = 1000,
        number_of_iterations: int = 400,
        theta: float = 0.0,
        print_summary: bool = False, 
        ax: Optional[plt.Axes] = None,
        label: Optional[str] = None,
        **plot_kwargs,
    ) -> Tuple[plt.Axes, MomentCurvatureResults]:
        """
        Runs `solve.monotonic_by_curvature` and plots the result.
        """
        result = self.solve.monotonic_by_curvature(
            axial_load=axial_load,
            max_curvature=max_curvature,
            number_of_points=number_of_points,
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
        ax.set_title("Monotonic Moment-Curvature (by Curvature)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if print_summary:
            self._print_plot_summary(result)
        
        return ax, result

    def plot_monotonic_by_strain(
        self,
        axial_load: float = 0.0,
        *,
        curvature_increment: float = 1e-6,
        max_steps: int = 2000,
        number_of_iterations: int = 400,
        theta: float = 0.0,
        print_summary: bool = False, 
        ax: Optional[plt.Axes] = None,
        label: Optional[str] = None,
        **plot_kwargs,
    ) -> Tuple[plt.Axes, MomentCurvatureResults]:
        """
        Runs `solve.monotonic_by_strain` and plots the result.
        """
        result = self.solve.monotonic_by_strain(
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
        ax.set_title("Monotonic Moment-Curvature (by Strain Limit)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if print_summary:
            self._print_plot_summary(result)
        
        return ax, result

    def plot_cyclic(
        self,
        axial_load: float = 0.0,
        *,
        protocol: Optional[TimeSeries] = None,
        number_of_points: int = 1000,
        curvature_scale_factor: float = None,
        scale_protocol_to_strain_limit: bool = True,
        number_of_iterations: int = 400,
        theta: float = 0.0,
        check_strain_limits: bool = True,
        print_summary: bool = False, 
        ax: Optional[plt.Axes] = None,
        label: Optional[str] = None,
        **plot_kwargs,
    ) -> Tuple[plt.Axes, MomentCurvatureResults]:
        """
        Runs `solve.cyclic` and plots the result.
        """
        result = self.solve.cyclic(
            axial_load=axial_load,
            protocol=protocol,
            number_of_points=number_of_points,
            curvature_scale_factor=curvature_scale_factor,
            scale_protocol_to_strain_limit=scale_protocol_to_strain_limit,
            number_of_iterations=number_of_iterations,
            theta=theta,
            check_strain_limits=check_strain_limits,
        )
        
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))
        if label is None:
            label = f"P = {result.axial_load:.2e} N, θ = {result.theta}°"
            
        ax.plot(result.curvatures, result.moments, label=label, **plot_kwargs)
        ax.set_xlabel("Curvature")
        ax.set_ylabel("Moment")
        title = "Cyclic Moment-Curvature"
        if check_strain_limits: title += " (Strain-Limited)"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if print_summary:
            self._print_plot_summary(result)
        
        return ax, result

    # --- Internal helper to avoid duplicating summary code ---
    def _print_plot_summary(self, result: MomentCurvatureResults):
        """Prints a formatted summary of analysis results."""
        print("\n--- Analysis Summary ---")
        print(f"Axial load: {result.axial_load:.3e} N")
        print(f"Peak curvature reached: {result.peak_curvature:.3e} (1/mm)")
        print(f"Peak moment reached: {result.peak_moment:.3e} (N-mm)")
        print(f"Controlling reason: {result.stop_reason}")
        
        print("\nFinal Max/Min Strains per Material:")
        
        final_fiber_state = result.fiber_history[-1, :, :]
        mat_tags = np.unique(final_fiber_state[:, 3])
        
        for tag in mat_tags:
            fibers_with_this_tag = final_fiber_state[
                final_fiber_state[:, 3] == tag
            ]
            # Handle case where no fibers exist for a tag (shouldn't happen, but safe)
            if fibers_with_this_tag.size == 0:
                continue
                
            strains_for_this_tag = fibers_with_this_tag[:, 5]
            
            max_comp_strain = np.min(strains_for_this_tag)
            max_tens_strain = np.max(strains_for_this_tag)
            
            print(f"  - Material Tag {int(tag)}:")
            print(f"    - Max Compressive Strain: {max_comp_strain:.3e}")
            print(f"    - Max Tensile Strain: {max_tens_strain:.3e}")
        
        print("------------------------\n")