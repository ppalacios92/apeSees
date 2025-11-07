"""Result data structures for section analysis."""

from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from ..utilities import AttrDict

import numpy as np
import matplotlib.pyplot as plt

@dataclass
class SectionResults:
    """
    Holds detailed section-level results at each analysis step.
    
    Attributes
    ----------
    deformation : np.ndarray, optional
        Section deformation history (n_steps, 4): [eps0, kappa_z, kappa_y, theta]
    force : np.ndarray, optional
        Section force/moment history (n_steps, 4): [P, Mz, My, T]
    tangent : np.ndarray, optional
        Section tangent stiffness history (n_steps, 4, 4)
    """
    deformation: Optional[np.ndarray] = None
    force: Optional[np.ndarray] = None
    tangent: Optional[np.ndarray] = None

@dataclass
class MomentCurvatureResults:
    """
        Results from a moment-curvature analysis.
        
        Attributes
        ----------
        axial_load : float
            Applied axial load.
        curvatures : np.ndarray
            Array of curvature values.
        moments : np.ndarray
            Array of moment values.
        fiber_history : np.ndarray
            Fiber state history with shape (n_steps, n_fibers, 6).
            Each fiber row contains: [y, z, area, mat_tag, stress, strain].
        theta : float, optional
            Rotation angle [degrees]. Default is 0.0.
        max_curvature : float, optional
            Maximum target curvature.
        converged : bool
            Whether the analysis converged successfully. Default is True.
        
        Examples
        --------
        >>> result = MomentCurvatureResults(
        ...     axial_load=-1000000.0,
        ...     curvatures=curvature_array,
        ...     moments=moment_array,
        ...     fiber_history=fiber_data
        ... )
        >>> print(f"Peak moment: {result.peak_moment:.2e} N·mm")
        >>> result.plot()
        >>> plt.show()
        """
    
    axial_load: float
    curvatures: np.ndarray
    moments: np.ndarray
    fiber_history: np.ndarray
    
    # section data
    section: Optional[SectionResults] = None
    
    # metadata
    theta: float = 0.0
    max_curvature: Optional[float] = None
    converged: bool = True
    
    # dictionary setup for extra data
    metadata: AttrDict = field(default_factory=AttrDict)
    
    def __post_init__(self):
        """Validate array shapes."""
        if len(self.curvatures) != len(self.moments):
            raise ValueError(
                f"Curvature and moment arrays must have same length: "
                f"{len(self.curvatures)} != {len(self.moments)}"
            )
        
        if self.fiber_history.ndim != 3 or self.fiber_history.shape[-1] != 6:
            raise ValueError(
                f"fiber_history must have shape (n_steps, n_fibers, 6), "
                f"got {self.fiber_history.shape}"
            )
        
        n_steps = len(self.curvatures)
        
        if self.section is not None:
            if self.section.deformation is not None:
                if self.section.deformation.shape != (n_steps, 4):
                    raise ValueError(
                        f"section.deformation must have shape ({n_steps}, 4), "
                        f"got {self.section.deformation.shape}"
                    )

            if self.section.force is not None:
                if self.section.force.shape != (n_steps, 4):
                    raise ValueError(
                        f"section.force must have shape ({n_steps}, 4), "
                        f"got {self.section.force.shape}"
                    )

            # Fixed typos: section_tangent -> self.section.tangent
            if self.section.tangent is not None:
                if self.section.tangent.shape != (n_steps, 4, 4):
                    raise ValueError(
                        f"section.tangent must have shape ({n_steps}, 4, 4), "
                        f"got {self.section.tangent.shape}"
                    )
    
    @property
    def peak_moment(self) -> float:
        """Maximum absolute moment value [N·mm]."""
        return float(np.max(np.abs(self.moments)))
    
    @property
    def peak_curvature(self) -> float:
        """Maximum absolute curvature value [1/mm]."""
        return float(np.max(np.abs(self.curvatures)))
    
    @property
    def num_steps(self) -> int:
        """Number of analysis steps."""
        return len(self.curvatures)
    
    @property
    def num_fibers(self) -> int:
        """Number of fibers in the section."""
        return self.fiber_history.shape[1]
    
    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: tuple[float, float] = (8, 5),
        label: Optional[str] = None,
        **kwargs
    ) -> plt.Axes:
        """
        Plot moment-curvature response.
        
        Parameters
        ----------
        ax : plt.Axes, optional
            Matplotlib axes. If None, creates new figure.
        figsize : tuple, optional
            Figure size in inches. Default is (8, 5).
        label : str, optional
            Plot label. If None, auto-generated from axial load.
        **kwargs
            Additional arguments passed to ax.plot().
        
        Returns
        -------
        plt.Axes
            The matplotlib axes.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        
        if label is None:
            label = f"P = {self.axial_load/1e6:.1f} MN"
        
        ax.plot(self.curvatures, self.moments, label=label, **kwargs)
        ax.set_xlabel("Curvature")
        ax.set_ylabel("Moment")
        ax.set_title("Moment-Curvature Response")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax
    
    def get_fiber_state(self, step: int) -> np.ndarray:
        """
        Get fiber state at a specific step.
        
        Parameters
        ----------
        step : int
            Analysis step index.
        
        Returns
        -------
        np.ndarray
            Fiber data with shape (n_fibers, 6): [y, z, area, mat_tag, stress, strain].
        """
        if step < 0 or step >= self.num_steps:
            raise IndexError(f"Step {step} out of range [0, {self.num_steps-1}]")
        
        return self.fiber_history[step]
    
    def axial_biaxial_stiffness(self, step: int) -> np.ndarray:
        """Return the 3×3 axial–bending stiffness (exclude torsion)."""
        if self.section.tangent is None:
            raise ValueError("section.tangent not stored")
        return self.section.tangent[step][np.ix_([0, 1, 2], [0, 1, 2])]
    
    def save(self, filepath: str) -> None:
            """
            Save results to a NumPy .npz file.
            """
            # Handle optional section data
            sec_deform = self.section.deformation if self.section and self.section.deformation is not None else np.array([])
            sec_force = self.section.force if self.section and self.section.force is not None else np.array([])
            sec_tangent = self.section.tangent if self.section and self.section.tangent is not None else np.array([])

            np.savez(
                filepath,
                axial_load=self.axial_load,
                curvatures=self.curvatures,
                moments=self.moments,
                fiber_history=self.fiber_history,
                
                # Save section data as individual arrays
                section_deformation=sec_deform,
                section_force=sec_force,
                section_tangent=sec_tangent,
                
                theta=self.theta,
                max_curvature=self.max_curvature or 0.0,
                converged=self.converged,
                metadata=self.metadata,
            )
    
    @classmethod
    def load(cls, filepath: str) -> "MomentCurvatureResults":
        
        """
        Load results from a NumPy .npz file.
        """
        
        data = np.load(filepath, allow_pickle=True)

        # --- Rebuild SectionResults object ---
        sec_deform = data["section_deformation"]
        if sec_deform.size == 0:
            sec_deform = None

        sec_force = data["section_force"]
        if sec_force.size == 0:
            sec_force = None

        sec_tangent = data["section_tangent"]
        if sec_tangent.size == 0:
            sec_tangent = None
        
        section_data = None
        if sec_deform is not None or sec_force is not None or sec_tangent is not None:
            section_data = SectionResults(
                deformation=sec_deform,
                force=sec_force,
                tangent=sec_tangent
            )
        # ---
        
        max_curv = float(data["max_curvature"])
        max_curv = max_curv if max_curv != 0.0 else None
        
        metadata_obj = data.get("metadata")
        metadata_dict = metadata_obj.item() if metadata_obj is not None else {}

        return cls(
            axial_load=float(data["axial_load"]),
            curvatures=data["curvatures"],
            moments=data["moments"],
            fiber_history=data["fiber_history"],
            section=section_data,  # <-- Pass the new object
            theta=float(data["theta"]),
            max_curvature=max_curv,
            converged=bool(data["converged"]),
            metadata=AttrDict(metadata_dict),
        )
    
    def __repr__(self) -> str:
            meta_count = len(self.metadata)
            meta_str = f", metadata_items={meta_count}" if meta_count > 0 else ""
            section_str = ", section_data=True" if self.section is not None else ""
            
            return (
            f"MomentCurvatureResults("
            f"P={self.axial_load/1e6:.2f} MN, "
            f"steps={self.num_steps}, "
            f"fibers={self.num_fibers}, "
            f"peak_M={self.peak_moment/1e9:.2f} GN·mm"
            f"{section_str}"  # <-- Added
            f"{meta_str})"    # <-- Fixed
        )