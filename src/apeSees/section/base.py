"""Base classes for structural sections."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Set

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from ..materials import Material  # <-- Import Material


class Section(ABC):
    """
    Abstract base class for all structural sections.
    
    All section types must implement:
    - build(): Define the section in OpenSees
    - plot_section(): Visualize the section geometry
    - get_materials(): Return all constituent materials
    """
    
    @abstractmethod
    def build(self, verbose: bool = False) -> int:
        """
        Build the section in OpenSees.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, prints section information. Default is False.
        
        Returns
        -------
        int
            The section tag.
        """
        pass
    
    @abstractmethod
    def plot_section(self, ax: plt.Axes | None = None, **kwargs) -> plt.Axes:
        """
        Plot the section geometry.
        
        Parameters
        ----------
        ax : plt.Axes, optional
            Matplotlib axes. If None, creates new figure.
        **kwargs
            Additional plotting arguments.
        
        Returns
        -------
        plt.Axes
            The matplotlib axes.
        """
        pass

    # --- NEW ABSTRACT METHOD ---
    @abstractmethod
    def get_materials(self) -> Set[Material]:
        """
        Return a set of all unique Material objects used in the section.
        
        Returns
        -------
        Set[Material]
            A set of the apeSees Material objects.
        """
        pass