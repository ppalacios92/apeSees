"""Neural/teacher-style moment-curvature analysis for fiber sections.

This class can be used for standard moment-curvature analysis or to
generate training data for neural sections by recording detailed
section-level history.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops

from .results import MomentCurvatureResults, SectionResults
from ..timeseries import LinearTimeSeries, ASCE41Protocol

if TYPE_CHECKING:
    from .base import Section
    from ..timeseries import TimeSeries


class NeuralMomentCurvatureTrainer:
    """
    Performs moment-curvature analysis on a fiber section.

    This class can optionally record detailed section-level data
    (deformation, force, tangent) for use in training neural models.
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
        Run moment-curvature analysis.
        
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

        self.section.material_core.build()
        self.section.material_cover.build()
        self.section.steel_material.build()
        section_tag = self.section.build()

        element_tag = 200
        theta_rad = np.radians(theta)
        ops.element(
            'zeroLengthSection',
            element_tag,
            BC_node, load_node,
            section_tag,
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
        
        # --- Conditional Allocation ---
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
            
        # Fiber history allocation
        probe = ops.eleResponse(element_tag, 'section', 'fiberData2')
        n_fib = len(probe) // 6
        fiber_history = np.zeros((n_steps, n_fib, 6))
        fiber_history[0, :, :] = np.array(probe, dtype=float).reshape(-1, 6)

        # --- Run analysis ---
        converged = True
        for i in range(n_steps):
            curvatures[i] = ops.nodeDisp(load_node, 5)
            moments[i] = ops.eleForce(element_tag, 5)

            # --- FIXED: Conditional Recording ---
            if record_section_data:
                sec_def = ops.eleResponse(element_tag, 'section', 'deformation')
                sec_force = ops.eleResponse(element_tag, 'section', 'force')
                sec_kt = ops.eleResponse(element_tag, 'section', 'stiffness')

                section_deformation[i, :] = np.array(sec_def, dtype=float)
                section_force[i, :] = np.array(sec_force, dtype=float)
                section_tangent[i, :, :] = np.array(sec_kt, dtype=float).reshape(4, 4)

            # Fiber history
            fdat = np.array(
                ops.eleResponse(element_tag, 'section', 'fiberData2'),
                dtype=float
            )
            fiber_history[i, :, :] = fdat.reshape(-1, 6)

            # Next step
            if i < n_steps - 1:
                if ops.analyze(1) != 0:
                    converged = False
                    # Truncate all arrays consistently
                    curvatures = curvatures[:i+1]
                    moments = moments[:i+1]
                    fiber_history = fiber_history[:i+1, :, :]
                    
                    # --- FIXED: Conditional Truncation ---
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
            section=section_results, # <-- This is now None or a SectionResults object
            theta=theta,
            max_curvature=max_curvature,
            converged=converged,
        )

    # --- FIXED: `plot` method ---
    def plot(
        self,
        axial_load: float = 0.0,
        number_of_points: int = 100,
        max_curvature: float = 0.004,
        number_of_iterations: int = 200, # <-- FIXED: Added missing param
        theta: float = 0.0,
        use_protocol: bool = False,
        protocol: Optional[TimeSeries] = None,
        ax: Optional[plt.Axes] = None,
        label: Optional[str] = None,
        **plot_kwargs,
    ) -> Tuple[plt.Axes, MomentCurvatureResults]:
        
        # Note: We intentionally do NOT pass record_section_data=True.
        # The plot() method doesn't need it, so we run `solve` in
        # its fastest, most lightweight mode.
        result = self.solve(
            axial_load=axial_load,
            number_of_points=number_of_points,
            max_curvature=max_curvature,
            number_of_iterations=number_of_iterations, # <-- FIXED: Pass param
            theta=theta,
            use_protocol=use_protocol,
            protocol=protocol,
        )

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        if label is None:
            label = f"P = {result.axial_load:.2e} N, θ = {result.theta}°"

        ax.plot(result.curvatures, result.moments, label=label, **plot_kwargs)
        
        # --- Added units to labels ---
        ax.set_xlabel("Curvature")
        ax.set_ylabel("Moment")
        ax.set_title("Moment-Curvature Response")
        ax.grid(True, alpha=0.3)
        ax.legend()

        return ax, result