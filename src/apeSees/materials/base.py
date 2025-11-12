"""Base material class for apeSees uniaxial materials."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import openseespy.opensees as ops

from .tester import UniaxialMaterialTester
from .results import MaterialTestResult


class Material:
    """
    Generic uniaxial material factory for OpenSees.

    Parameters
    ----------
    mat_type : str
        The OpenSees uniaxial material type (e.g., 'Steel02', 'Concrete02').
    tag : int
        Unique material tag.
    *params : Any
        Material parameters in the exact positional order required by OpenSees.
    max_tensile_strain : float, optional
        The maximum tensile strain this material can withstand before failure.
        Used by solvers like `solve_by_strain`. Default is 1e10 (no limit).
    max_compressive_strain : float, optional
        The maximum compressive strain (negative value) this material
        can withstand before failure. Default is -1e10 (no limit).
    
    Examples
    --------
    >>> steel = Material(
    ...     "Steel02", 1, 420.0, 200000.0, 0.01,
    ...     max_tensile_strain=0.05
    ... )
    >>> core_concrete = Material(
    ...     "Concrete02", 2, -30.0, -0.002, -20.0, -0.006,
    ...     max_compressive_strain=-0.006
    ... )
    """

    def __init__(
        self, 
        mat_type: str, 
        tag: int, 
        *params: Any,
        # --- NEW KWARGS ---
        max_tensile_strain: Optional[float] = None,
        max_compressive_strain: Optional[float] = None
    ):
        self.mat_type: str = str(mat_type)
        self.tag: int = int(tag)
        self.params: list[Any] = list(params)

        if not self.params:
            raise ValueError(
                f"No parameters provided for material '{self.mat_type}'. "
                f"Expected positional arguments in OpenSees order."
            )
        
        self.tester: UniaxialMaterialTester = UniaxialMaterialTester(material_object=self)
        
        # --- NEW ATTRIBUTES ---
        # Set "infinite" limits if none are provided
        self.max_tensile_strain: float = max_tensile_strain if max_tensile_strain is not None else 1e10
        self.max_compressive_strain: float = max_compressive_strain if max_compressive_strain is not None else -1e10
        # --- END NEW ATTRIBUTES ---

    def build(self, verbose: bool = False) -> int:
        """
        Define the material in OpenSees and return its tag.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, prints the OpenSees command being executed. Default is False.
        
        Returns
        -------
        int
            The material tag.
        """
        ops.uniaxialMaterial(self.mat_type, self.tag, *self.params)
        
        if verbose:
            params_str = ", ".join(map(str, self.params))
            print(f'ops.uniaxialMaterial("{self.mat_type}", {self.tag}, {params_str})')
        
        return self.tag
    
    def cyclic_tester(
        self,
        *,
        max_strain: float,
        protocol_type: str = 'asce41',
        number_of_points: int = 1000,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[float, float] = (10, 6),
        show_backbone: bool = True,
        hysteresis_color: str = '#000077',
        backbone_color: str = 'black',
        **kwargs
    ) -> Tuple[plt.Axes, MaterialTestResult]:
        """
        Plot cyclic response with optional backbone overlay.
        
        (Method implementation is unchanged)
        """
        from ..timeseries import ASCE41Protocol, ModifiedATC24Protocol, FEMA461Protocol
        
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        
        # Select protocol
        protocol_map = {
            'asce41': ASCE41Protocol,
            'atc24': ModifiedATC24Protocol,
            'fema461': FEMA461Protocol,
        }
        
        protocol_type_lower = protocol_type.lower()
        if protocol_type_lower not in protocol_map:
            raise ValueError(
                f"Unknown protocol_type '{protocol_type}'. "
                f"Choose from: {list(protocol_map.keys())}"
            )
        
        protocol_class = protocol_map[protocol_type_lower]
        protocol = protocol_class(tag=1, max_disp=max_strain)
        
        # Plot hysteresis and get result
        ax, result = self.tester.plot(
            time_series=protocol,
            number_of_points=number_of_points,
            ax=ax,
            label=f'Hysteretic Response - {self.name}',
            linestyle='-',
            color=hysteresis_color,
            linewidth=1.0,
            title='Uniaxial Hysteretic Response',
        )
        
        # Overlay backbone if requested
        if show_backbone:
            ax, backbone_result = self.tester.plot_backbone(
                strain_max=max_strain,
                strain_min=-max_strain,
                ax=ax,
                linestyle='--',
                color=backbone_color,
                linewidth=3,
                label='Backbone',
                title=None,  # Don't override title
            )
        
        ax.legend()
        
        return ax, result

    def backbone_tester(
        self,
        *,
        max_strain: float,
        number_of_points: int = 100,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[float, float] = (10, 6),
        **kwargs
    ) -> Tuple[plt.Axes, MaterialTestResult]:
        """
        Plot monotonic backbone curve.
        
        (Method implementation is unchanged)
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        
        # Set defaults if not provided
        plot_kwargs = {
            'linestyle': '-',
            'color': 'black',
            'linewidth': 2,
            'label': f'Backbone - {self.name}'
        }
        plot_kwargs.update(kwargs)
        
        ax, result = self.tester.plot_backbone(
            strain_max=max_strain,
            strain_min=-max_strain,
            number_of_points=number_of_points,
            ax=ax,
            **plot_kwargs
        )
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax, result
    
    @property
    def name(self) -> str:
        """
        Material name for display purposes.
        """
        return f"{self.mat_type} - matTag:{self.tag}"
    
    def __repr__(self) -> str:
        return f"Material({self.mat_type!r}, tag={self.tag}, params={self.params})"