"""Neural/teacher-style moment-curvature analysis for fiber sections.

This is a drop-in variant of `MomentCurvature` that, in addition to the usual
curvature–moment and fiber history, also records:

- section_deformation: (n_steps, 4) = [eps0, kappa_z, kappa_y, theta_t]
- section_force:       (n_steps, 4) = [P, Mz, My, T]
- section_tangent:     (n_steps, 4, 4)

So it can be used to generate training data for neural sections.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops

from .results import MomentCurvatureResults
from ..timeseries import LinearTimeSeries, ASCE41Protocol

if TYPE_CHECKING:
    from .base import Section
    from ..timeseries import TimeSeries


class NeuralMomentCurvatureTrainer:
    """
    Same usage as `MomentCurvature`, but returns richer results.
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
    ) -> MomentCurvatureResults:
        # --- OpenSees model setup (same as your original) ---
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)

        # nodes
        BC_node = 1
        load_node = 2
        ops.node(BC_node, 0.0, 0.0, 0.0)
        ops.node(load_node, 0.0, 0.0, 0.0)

        # BCs
        ops.fix(BC_node, *[1, 1, 1, 1, 1, 1])
        ops.fix(load_node, *[0, 1, 1, 1, 0, 1])  # ux & rz free

        # materials
        self.section.material_core.build()
        self.section.material_cover.build()
        self.section.steel_material.build()

        # build section
        section_tag = self.section.build()

        # element with rotation
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

        # (optional) fiber recorder, keep same as original
        ops.recorder(
            'Element', '-time',
            '-file', 'fiberData.out',
            '-ele', element_tag,
            'section', 'fiberData'
        )

        # axial load
        ts_ax = 500
        ops.timeSeries('Constant', ts_ax)
        pat_ax = 600
        ops.pattern('Plain', pat_ax, ts_ax)
        ops.load(load_node, axial_load, 0, 0, 0, 0, 0)

        # analysis for axial load
        ops.integrator('LoadControl', 0.0)
        ops.system('SuperLU')
        ops.test('NormUnbalance', 1e-6, number_of_iterations, 0)
        ops.numberer('Plain')
        ops.constraints('Transformation')
        ops.algorithm('KrylovNewton')
        ops.analysis('Static')
        ops.analyze(1)

        # time series for MC
        timeseries_tag = 501
        if use_protocol and protocol is not None:
            protocol.build()
            timeseries_tag = protocol.tag
        elif use_protocol:
            ASCE41Protocol(tag=timeseries_tag, max_disp=1.0).build()
        else:
            LinearTimeSeries(timeseries_tag, factor=1.0).build()

        # pattern for MC
        pat_mc = 601
        ops.pattern('Plain', pat_mc, timeseries_tag)
        ops.sp(load_node, 5, max_curvature)  # rotation about z

        # integrator for steps
        dκ = 1.0 / number_of_points
        ops.integrator('LoadControl', dκ)

        # --- allocate results ---
        n_steps = number_of_points + 1
        curvatures = np.zeros(n_steps)
        moments = np.zeros(n_steps)

        # section-level arrays (this is the extra part)
        section_deformation = np.zeros((n_steps, 4))
        section_force = np.zeros((n_steps, 4))
        section_tangent = np.zeros((n_steps, 4, 4))

        # fiber history
        probe = ops.eleResponse(element_tag, 'section', 'fiberData2')
        n_fib = len(probe) // 6
        fiber_history = np.zeros((n_steps, n_fib, 6))
        fiber_history[0, :, :] = np.array(probe, dtype=float).reshape(-1, 6)

        # --- run analysis ---
        converged = True
        for i in range(n_steps):
            # kappa (from node rotation)
            curvatures[i] = ops.nodeDisp(load_node, 5)
            # moment (reaction)
            moments[i] = ops.eleForce(element_tag, 5)

            # section data (the new part)
            sec_def = ops.eleResponse(element_tag, 'section', 'deformation')
            sec_force = ops.eleResponse(element_tag, 'section', 'force')
            sec_kt = ops.eleResponse(element_tag, 'section', 'stiffness')

            section_deformation[i, :] = np.array(sec_def, dtype=float)
            section_force[i, :] = np.array(sec_force, dtype=float)
            section_tangent[i, :, :] = np.array(sec_kt, dtype=float).reshape(4, 4)

            # fiber history
            fdat = np.array(
                ops.eleResponse(element_tag, 'section', 'fiberData2'),
                dtype=float
            )
            fiber_history[i, :, :] = fdat.reshape(-1, 6)

            # next step
            if i < n_steps - 1:
                if ops.analyze(1) != 0:
                    converged = False
                    # truncate all arrays consistently
                    curvatures = curvatures[:i+1]
                    moments = moments[:i+1]
                    fiber_history = fiber_history[:i+1, :, :]
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
            section_deformation=section_deformation,
            section_force=section_force,
            section_tangent=section_tangent,
            theta=theta,
            max_curvature=max_curvature,
            converged=converged,
        )

    # optional: same plotting helpers so interface matches
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
        ax.set_xlabel("Curvature [1/mm]")
        ax.set_ylabel("Moment [N·mm]")
        ax.set_title("Moment-Curvature Response (Neural)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        return ax, result
