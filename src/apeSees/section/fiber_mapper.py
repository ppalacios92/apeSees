"""Utility to map and plot fiber section data."""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops
from matplotlib.patches import Rectangle

if TYPE_CHECKING:
    from .base import Section


class FiberMapper:
    """
    Extracts, stores, and plots the static fiber map for a section.
    
    This class runs a minimal "dummy" analysis upon first request to
    build the section and use `eleResponse` to get the `fiberData2`
    output, which includes coordinates, area, and material tags.
    
    The results are cached internally, so the dummy analysis runs only once.
    
    Parameters
    ----------
    section : Section
        The fiber section to map.
        
    Examples
    --------
    >>> # Assuming 'section' is an existing Section object
    >>> mapper = section.fiber_map
    >>>
    >>> # Get the fiber coordinates
    >>> y_coords = mapper.y
    >>> z_coords = mapper.z
    >>>
    >>> # Get the material tags
    >>> tags = mapper.mat_tags
    >>> print(f"Found {mapper.n_fibers} fibers.")
    >>>
    >>> # Plot the fiber layout
    >>> fig, ax = plt.subplots()
    >>> mapper.plot(ax=ax)
    >>> plt.show()
    """
    
    def __init__(self, section: Section):
        self.section: Section = section
        
        # --- Internal cache ---
        self._mapped: bool = False
        self._n_fibers: int = 0
        self._y: Optional[np.ndarray] = None
        self._z: Optional[np.ndarray] = None
        self._areas: Optional[np.ndarray] = None
        self._mat_tags: Optional[np.ndarray] = None
        self._fiber_tags: Optional[np.ndarray] = None # This is just the index

    def _run_mapper(self) -> None:
        """
        Runs the dummy OpenSees model to extract fiber data.
        """
        if self._mapped:
            return

        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)
        
        # Minimal nodes/element
        ops.node(1, 0.0, 0.0, 0.0)
        ops.node(2, 0.0, 0.0, 0.0)
        ops.fix(1, *[1, 1, 1, 1, 1, 1])

        # Build materials (needed by section)
        self.section.material_core.build()
        self.section.material_cover.build()
        self.section.steel_material.build()

        # Build section
        section_tag = self.section.build()
        
        # Build element
        element_tag = 1
        ops.element('zeroLengthSection', element_tag, 1, 2, section_tag)

        # --- This is the key ---
        # Get the fiber data at time 0
        try:
            probe = ops.eleResponse(element_tag, 'section', 'fiberData2')
        except Exception as e:
            ops.wipe()
            raise RuntimeError(
                "Failed to get 'fiberData2' from section. "
                "Ensure the section is correctly defined."
            ) from e
            
        ops.wipe()

        # --- Process and store the data ---
        # fiberData2 returns a flat list: [y, z, A, matTag, sig, eps]
        data = np.array(probe, dtype=float).reshape(-1, 6)
        
        self._n_fibers = data.shape[0]
        self._y = data[:, 0]
        self._z = data[:, 1]
        self._areas = data[:, 2]
        self._mat_tags = data[:, 3].astype(int)
        self._fiber_tags = np.arange(self._n_fibers) # Implicit fiber tag
        
        self._mapped = True

    def get_index_at_coord(self, y: float, z: float) -> int:
            """
            Finds the row index (implicit tag) of the fiber
            closest to the given (y, z) coordinates.
            
            Parameters
            ----------
            y : float
                Target y-coordinate [mm].
            z : float
                Target z-coordinate [mm].
                
            Returns
            -------
            int
                The row index of the closest fiber.
            """
            # Ensure data is loaded
            if not self._mapped:
                self._run_mapper()
                
            # Calculate squared Euclidean distance (faster than sqrt)
            dist_sq = (self.y - y)**2 + (self.z - z)**2
            
            # Find the index of the minimum distance
            index = np.argmin(dist_sq)
            
            return int(index)

    # --- Public Properties ---
    
    @property
    def n_fibers(self) -> int:
        """Number of fibers in the section."""
        if not self._mapped:
            self._run_mapper()
        return self._n_fibers

    @property
    def y(self) -> np.ndarray:
        """Fiber y-coordinates [mm]."""
        if not self._mapped:
            self._run_mapper()
        return self._y

    @property
    def z(self) -> np.ndarray:
        """Fiber z-coordinates [mm]."""
        if not self._mapped:
            self._run_mapper()
        return self._z

    @property
    def areas(self) -> np.ndarray:
        """Fiber areas [mm^2]."""
        if not self._mapped:
            self._run_mapper()
        return self._areas

    @property
    def mat_tags(self) -> np.ndarray:
        """Fiber material tags."""
        if not self._mapped:
            self._run_mapper()
        return self._mat_tags

    @property
    def fiber_tags(self) -> np.ndarray:
        """Fiber index (0 to n_fibers-1)."""
        if not self._mapped:
            self._run_mapper()
        return self._fiber_tags

    # --- Plotting Method ---
    
    def plot(
            self,
            ax: Optional[plt.Axes] = None,
            scale_by_area: bool = True,
            min_size: float = 1.0,
            max_size: float = 500.0,
            show_outline: bool = True,
            show_fiber_tags: bool = False,
            fiber_tag_fontsize: float = 6.0,
            figsize: Optional[tuple] = (7,6),
            **kwargs
        ) -> plt.Axes:
            """
            Plots the fiber map using `scatter`, grouped by material for a legend.
            
            Parameters
            ----------
            ax : plt.Axes, optional
                Matplotlib axes. If None, creates new figure.
            scale_by_area : bool, optional
                If True, scales marker size by fiber area. Default is True.
            min_size : float, optional
                The display size (in points^2) for the smallest fiber. Default is 1.0.
            max_size : float, optional
                The display size (in points^2) for the largest fiber. Default is 500.0.
            show_outline : bool, optional
                If True, attempts to draw the section outline. Default is True.
            show_fiber_tags : bool, optional
                If True, annotates each fiber with its row index. Default is False.
            fiber_tag_fontsize : float, optional
                Font size for the fiber tag annotations. Default is 6.0.
            **kwargs
                Additional arguments passed to ax.scatter().
            
            Returns
            -------
            plt.Axes
                The matplotlib axes.
            """
            if ax is None:
                _, ax = plt.subplots(figsize=figsize)
                
            # Get data (this will trigger _run_mapper if needed)
            y, z = self.y, self.z
            areas, tags = self.areas, self.mat_tags

            # --- Marker Size Logic (Unchanged) ---
            if scale_by_area and self.n_fibers > 1:
                min_area = np.min(areas)
                max_area = np.max(areas)
                
                if np.isclose(min_area, max_area):
                    s = np.full_like(areas, (min_size + max_size) / 2)
                else:
                    norm_areas = (areas - min_area) / (max_area - min_area)
                    s = norm_areas * (max_size - min_size) + min_size
                    
            elif scale_by_area: # Only 1 fiber
                s = (min_size + max_size) / 2
            else:
                s = kwargs.pop('s', 30)
                # If not scaling, 's' is a single value, so expand it to an array
                if isinstance(s, (int, float)):
                    s = np.full_like(areas, s)
            
            # --- NEW: Plot by group for legend ---
            unique_tags = np.unique(tags)
            # Get a standard colormap to pick from
            cmap = plt.get_cmap("tab10") 
            colors = {tag: cmap(i) for i, tag in enumerate(unique_tags)}

            for tag in unique_tags:
                mask = (tags == tag)
                ax.scatter(
                    y[mask], z[mask], 
                    s=s[mask], 
                    color=colors[tag], 
                    label=f"Mat Tag: {tag}", 
                    **kwargs
                )
            
            ax.legend(title="Materials", loc='best')
            # --- End new plot logic ---

            # --- NEW: Show fiber tags ---
            if show_fiber_tags:
                for i in range(self.n_fibers):
                    ax.text(
                        y[i], z[i], str(i), 
                        fontsize=fiber_tag_fontsize, 
                        ha='center', 
                        va='center', 
                        zorder=5, # Ensure text is on top
                        color='k' # Use black for readability
                    )
            # --- End show tags logic ---
            
            # Add section outline
            if show_outline and hasattr(self.section, 'B') and hasattr(self.section, 'H'):
                B, H = self.section.B, self.section.H
                ax.add_patch(Rectangle(
                    (-B/2, -H/2), B, H,
                    facecolor='none', edgecolor='k',
                    linestyle='--', linewidth=1.0, zorder=1
                ))
            
            ax.set_aspect('equal')
            ax.set_xlabel("y [mm]")
            ax.set_ylabel("z [mm]")
            ax.set_title(f"Section Fiber Map ({self.n_fibers} fibers)")
            ax.grid(True, alpha=0.3)
            
            return ax