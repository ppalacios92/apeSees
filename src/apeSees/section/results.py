"""Result data structures for section analysis."""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import warnings

from ..utilities import AttrDict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

if TYPE_CHECKING:
    from .base import Section # This import is not in the original, but good practice
    pass

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
        section : SectionResults, optional
            Detailed section-level results (if recorded).
        theta : float, optional
            Rotation angle [degrees]. Default is 0.0.
        max_curvature : float, optional
            Maximum target curvature.
        converged : bool
            Whether the analysis converged successfully. Default is True.
        stop_reason : str, optional
            Reason the analysis stopped (e.g., "Tensile limit reached").
        metadata : AttrDict, optional
            Additional metadata about the test.
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
    stop_reason: Optional[str] = None
    
    # dictionary setup for extra data
    metadata: AttrDict = field(default_factory=AttrDict)

    # --- INTERNAL CACHE FOR PROPERTIES ---
    _yield_properties: Optional[Tuple[float, float]] = field(init=False, repr=False, default=None)
    
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
                        f"section.deformation shape must be ({n_steps}, 4), "
                        f"got {self.section.deformation.shape}"
                    )

            if self.section.force is not None:
                if self.section.force.shape != (n_steps, 4):
                    raise ValueError(
                        f"section.force shape must be ({n_steps}, 4), "
                        f"got {self.section.force.shape}"
                    )

            if self.section.tangent is not None:
                if self.section.tangent.shape != (n_steps, 4, 4):
                    raise ValueError(
                        f"section.tangent shape must be ({n_steps}, 4, 4), "
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

    @property
    def initial_stiffness(self) -> float:
        """
        Calculates the initial elastic stiffness (EI) from the second step.
        Returns 0.0 if calculation fails.
        """
        if self.num_steps < 2:
            return 0.0
        
        # Use second step (index 1) to avoid divide-by-zero at (0,0)
        dM = self.moments[1] - self.moments[0]
        dK = self.curvatures[1] - self.curvatures[0]
        
        if abs(dK) < 1e-15:
            return 0.0
            
        return dM / dK

    # --- NEW PROPERTY: DUCTILITY ---
    @property
    def ductility(self) -> float:
        """
        Calculates curvature ductility (mu_phi = phi_u / phi_y).
        
        Requires `get_yield_properties()` to be called first to define
        the yield point. Returns np.nan if yield properties are not set.
        """
        if self._yield_properties is None:
            warnings.warn(
                "Ductility requires a yield point. Call "
                "`get_yield_properties(mat_tag, strain)` first."
            )
            return np.nan
            
        yield_curvature = self._yield_properties[0]
        ultimate_curvature = self.peak_curvature
        
        if abs(yield_curvature) < 1e-15:
            return np.nan
            
        return ultimate_curvature / abs(yield_curvature)

    
    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: tuple[float, float] = (8, 5),
        label: Optional[str] = None,
        **kwargs
    ) -> plt.Axes:
        """
        Plot moment-curvature response.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        
        if label is None:
            label = f"P = {self.axial_load:.1f}"
        
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
        
        Returns
        -------
        np.ndarray
            Shape (n_fibers, 6) with:
            [y, z, area, mat_tag, stress, strain]
        """
        if step < 0: # Handle negative indexing
            step = self.num_steps + step
            
        if not (0 <= step < self.num_steps):
            raise IndexError(f"Step {step} out of range [0, {self.num_steps-1}]")
        
        return self.fiber_history[step]

    def plot_strain_envelope(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: tuple[float, float] = (8, 5),
        label_max: str = "Max Strain (Tension)",
        label_min: str = "Min Strain (Compression)",
        max_kwargs: Optional[Dict[str, Any]] = None,
        min_kwargs: Optional[Dict[str, Any]] = None,
    ) -> plt.Axes:
        """
        Plots the maximum and minimum fiber strains versus curvature.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        
        if max_kwargs is None: max_kwargs = {}
        if min_kwargs is None: min_kwargs = {}
        max_kwargs.setdefault('label', label_max)
        min_kwargs.setdefault('label', label_min)
        min_kwargs.setdefault('linestyle', '--')

        all_strains = self.fiber_history[:, :, 5]
        max_strains = np.max(all_strains, axis=1)
        min_strains = np.min(all_strains, axis=1)
        
        ax.plot(self.curvatures, max_strains, **max_kwargs)
        
        if 'color' not in min_kwargs:
            min_kwargs['color'] = ax.get_lines()[-1].get_color()
        
        ax.plot(self.curvatures, min_strains, **min_kwargs)
        
        ax.set_xlabel("Curvature")
        ax.set_ylabel("Strain")
        ax.set_title("Strain Envelope vs. Curvature")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax

    def plot_fiber_strains(
        self,
        fiber_indices: List[int],
        ax: Optional[plt.Axes] = None,
        figsize: tuple[float, float] = (8, 5),
        **kwargs
    ) -> plt.Axes:
        """
        Plots strain vs. curvature for specific fiber indices.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        
        all_strains = self.fiber_history[:, :, 5]
        
        for idx in fiber_indices:
            if not (isinstance(idx, int) and -self.num_fibers <= idx < self.num_fibers):
                print(f"Warning: Skipping invalid fiber index {idx}. Max index is {self.num_fibers - 1}")
                continue
            
            real_idx = idx if idx >= 0 else self.num_fibers + idx
            fiber_strain_history = all_strains[:, idx]
            
            plot_kwargs = kwargs.copy()
            plot_kwargs.setdefault('label', f"Fiber {real_idx}")
            
            ax.plot(self.curvatures, fiber_strain_history, **plot_kwargs)

        ax.set_xlabel("Curvature")
        ax.set_ylabel("Fiber Strain")
        ax.set_title("Fiber Strain vs. Curvature")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax

    # --- NEW METHOD: PLOT STRESS-STRAIN ---
    def plot_stress_strain(
        self,
        fiber_indices: List[int],
        ax: Optional[plt.Axes] = None,
        figsize: tuple[float, float] = (8, 5),
        **kwargs
    ) -> plt.Axes:
        """
        Plots stress vs. strain for specific fiber indices.
        
        Parameters
        ----------
        fiber_indices : List[int]
            A list of integer indices for the fibers to plot.
        ax : plt.Axes, optional
            Existing matplotlib axes to plot on.
        figsize : tuple, optional
            Figure size if a new plot is created.
        **kwargs
            Keyword arguments passed to ax.plot() for all lines.
        
        Returns
        -------
        plt.Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        
        all_strains = self.fiber_history[:, :, 5]
        all_stresses = self.fiber_history[:, :, 4]
        
        for idx in fiber_indices:
            if not (isinstance(idx, int) and -self.num_fibers <= idx < self.num_fibers):
                print(f"Warning: Skipping invalid fiber index {idx}. Max index is {self.num_fibers - 1}")
                continue
            
            real_idx = idx if idx >= 0 else self.num_fibers + idx
            strains = all_strains[:, idx]
            stresses = all_stresses[:, idx]
            
            plot_kwargs = kwargs.copy()
            plot_kwargs.setdefault('label', f"Fiber {real_idx}")
            
            ax.plot(strains, stresses, **plot_kwargs)

        ax.set_xlabel("Strain")
        ax.set_ylabel("Stress")
        ax.set_title("Fiber Stress-Strain History")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax

    def plot_section_state(
        self, 
        step: int,
        plot_by: str = 'strain',
        ax: Optional[plt.Axes] = None,
        figsize: tuple[float, float] = (6, 6),
        cmap: str = 'RdBu_r',
        title: Optional[str] = None,
        marker_scale: float = 1.0,
        **kwargs
    ) -> plt.Axes:
        """
        Plots the cross-section state (strain or stress) at a specific step.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        
        state = self.get_fiber_state(step)
        
        y_coords = state[:, 0]
        z_coords = state[:, 1]
        areas = state[:, 2]
        
        if plot_by.lower() == 'strain':
            values = state[:, 5] # strain
            cbar_label = "Strain"
        elif plot_by.lower() == 'stress':
            values = state[:, 4] # stress
            cbar_label = "Stress"
        else:
            raise ValueError(f"plot_by must be 'strain' or 'stress', got {plot_by}")
        
        min_size = 1.0
        sizes = (np.sqrt(areas) * marker_scale) + min_size
        
        norm = Normalize(vmin=np.min(values), vmax=np.max(values))
        
        sc = ax.scatter(y_coords, z_coords, c=values, s=sizes, cmap=cmap, norm=norm, **kwargs)
        
        plt.colorbar(sc, ax=ax, label=cbar_label, orientation='vertical')
        
        if title is None:
            title = (
                f"Section {cbar_label} State at Step {step}\n"
                f"Curvature = {self.curvatures[step]:.2e}, Moment = {self.moments[step]:.2e}"
            )
        
        ax.set_title(title)
        ax.set_xlabel("Y-coordinate")
        ax.set_ylabel("Z-coordinate")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        
        return ax

    def get_curvature_at_strain(
        self, 
        material_tag: int, 
        strain_value: float,
        search_from_start: bool = True
    ) -> Optional[float]:
        """
        Finds the curvature when a material first reaches a target strain.
        """
        
        mat_mask = self.fiber_history[0, :, 3] == material_tag
        if not np.any(mat_mask):
            warnings.warn(f"Material tag {material_tag} not found in results.")
            return None
            
        material_strains = self.fiber_history[:, mat_mask, 5]
        
        if strain_value >= 0: # Tensile strain
            envelope = np.max(material_strains, axis=1)
            condition = envelope >= strain_value
        else: # Compressive strain
            envelope = np.min(material_strains, axis=1)
            condition = envelope <= strain_value
            
        if not np.any(condition):
            warnings.warn(f"Strain value {strain_value} was not reached for material {material_tag}.")
            return None
        
        if search_from_start:
            step_index = np.argmax(condition)
        else:
            step_index = len(condition) - 1 - np.argmax(condition[::-1])

        return self.curvatures[step_index]

    # --- NEW METHOD: GET YIELD PROPERTIES ---
    def get_yield_properties(
        self,
        material_tag: int,
        yield_strain: float,
    ) -> Tuple[float, float]:
        """
        Calculates and caches the yield curvature and moment.
        
        Yield is defined as the point when the specified material 
        first reaches the target `yield_strain`.
        
        Parameters
        ----------
        material_tag : int
            The material tag of the yielding material (e.g., steel rebar).
        yield_strain : float
            The strain at which yield occurs.
            
        Returns
        -------
        Tuple[float, float]
            (yield_curvature, yield_moment)
        """
        # Find the curvature at yield
        yield_curvature = self.get_curvature_at_strain(
            material_tag=material_tag,
            strain_value=yield_strain
        )
        
        if yield_curvature is None:
            warnings.warn(f"Yield strain {yield_strain} not reached for mat {material_tag}.")
            self._yield_properties = (np.nan, np.nan)
            return (np.nan, np.nan)

        # Interpolate the moment at that curvature
        # Use np.interp for this
        yield_moment = np.interp(
            yield_curvature, 
            self.curvatures, 
            self.moments
        )
        
        # Cache and return
        self._yield_properties = (yield_curvature, yield_moment)
        print(f"Yield properties set: (φ_y = {yield_curvature:.3e}, M_y = {yield_moment:.3e})")
        return self._yield_properties


    def axial_biaxial_stiffness(self, step: int) -> np.ndarray:
        """Return the 3×3 axial–bending stiffness (exclude torsion)."""
        if self.section is None or self.section.tangent is None:
            raise ValueError("section.tangent not stored")
        return self.section.tangent[step][np.ix_([0, 1, 2], [0, 1, 2])]
    
    def save(self, filepath: str) -> None:
        """
        Save results to a NumPy .npz file.
        """
        sec_deform = self.section.deformation if self.section and self.section.deformation is not None else np.array([])
        sec_force = self.section.force if self.section and self.section.force is not None else np.array([])
        sec_tangent = self.section.tangent if self.section and self.section.tangent is not None else np.array([])

        np.savez(
            filepath,
            axial_load=self.axial_load,
            curvatures=self.curvatures,
            moments=self.moments,
            fiber_history=self.fiber_history,
            
            section_deformation=sec_deform,
            section_force=sec_force,
            section_tangent=sec_tangent,
            
            theta=self.theta,
            max_curvature=self.max_curvature or 0.0,
            converged=self.converged,
            stop_reason=self.stop_reason or '',
            metadata=self.metadata,
        )
    
    @classmethod
    def load(cls, filepath: str) -> "MomentCurvatureResults":
        
        """
        Load results from a NumPy .npz file.
        """
        
        data = np.load(filepath, allow_pickle=True)

        sec_deform = data["section_deformation"]
        if sec_deform.size == 0: sec_deform = None
        sec_force = data["section_force"]
        if sec_force.size == 0: sec_force = None
        sec_tangent = data["section_tangent"]
        if sec_tangent.size == 0: sec_tangent = None
        
        section_data = None
        if sec_deform is not None or sec_force is not None or sec_tangent is not None:
            section_data = SectionResults(
                deformation=sec_deform,
                force=sec_force,
                tangent=sec_tangent
            )
        
        max_curv = float(data["max_curvature"])
        max_curv = max_curv if max_curv != 0.0 else None
        
        metadata_obj = data.get("metadata")
        metadata_dict = metadata_obj.item() if metadata_obj is not None else {}

        stop_reason = data.get("stop_reason")
        if stop_reason is not None:
            stop_reason = str(stop_reason) if str(stop_reason) else None
        
        return cls(
            axial_load=float(data["axial_load"]),
            curvatures=data["curvatures"],
            moments=data["moments"],
            fiber_history=data["fiber_history"],
            section=section_data,
            theta=float(data["theta"]),
            max_curvature=max_curv,
            converged=bool(data["converged"]),
            stop_reason=stop_reason,
            metadata=AttrDict(metadata_dict),
        )
    
    def __repr__(self) -> str:
        meta_count = len(self.metadata)
        meta_str = f", metadata_items={meta_count}" if meta_count > 0 else ""
        section_str = ", section_data=True" if self.section is not None else ""
        stop_str = f", stop_reason={self.stop_reason!r}" if self.stop_reason else ""
        
        # --- ADDED EI TO REPR ---
        return (
            f"MomentCurvatureResults("
            f"P={self.axial_load:.2f}, "
            f"steps={self.num_steps}, "
            f"fibers={self.num_fibers}, "
            f"EI={self.initial_stiffness:.2f}, "
            f"peak_M={self.peak_moment:.2f}"
            f"{section_str}"
            f"{stop_str}"
            f"{meta_str})"
        )