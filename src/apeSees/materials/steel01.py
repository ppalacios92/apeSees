from __future__ import annotations
from typing import Any, Optional

# --- Use a relative import ---
from .base import Material

class Steel01(Material):
    """
    Represents the OpenSees Steel01 uniaxial material.

    This is a bilinear steel material with kinematic hardening
    and optional isotropic hardening.

    OpenSees Documentation Parameters:
    matTag, Fy, E0, b, <a1, a2, a3, a4>
    
    Parameters
    ----------
    tag : int
        Unique material tag.
    fy : float
        Yield strength.
    e : float
        Initial elastic tangent (Young's modulus).
    b : float
        Strain hardening ratio (E_sh / E).
    a1 : float, optional
        Isotropic hardening parameter (Default: None)
    a2 : float, optional
        Isotropic hardening parameter (Default: None)
    a3 : float, optional
        Isotropic hardening parameter (Default: None)
    a4 : float, optional
        Isotropic hardening parameter (Default: None)
    max_tensile_strain : float, optional
        Tensile strain limit. Default is 1e10 (no limit).
    max_compressive_strain : float, optional
        Compressive strain limit. Default is -1e10 (no limit).
    """
    def __init__(self,
                 tag: int,
                 fy: float,
                 e: float,
                 b: float,
                 a1: Optional[float] = None,
                 a2: Optional[float] = None,
                 a3: Optional[float] = None,
                 a4: Optional[float] = None,
                 *,
                 # --- NEW KWARGS ---
                 max_tensile_strain: Optional[float] = None,
                 max_compressive_strain: Optional[float] = None):
        
        # --- Parameter list logic (unchanged) ---
        mat_params = [fy, e, b]
        
        opt_params = [a1, a2, a3, a4]
        
        if any(p is not None for p in opt_params):
            a1 = a1 if a1 is not None else 0.0
            a2 = a2 if a2 is not None else 1.0
            a3 = a3 if a3 is not None else 0.0
            a4 = a4 if a4 is not None else 1.0
            mat_params.extend([a1, a2, a3, a4])

        # --- Call the parent class __init__ (MODIFIED) ---
        super().__init__(
            "Steel01", 
            tag, 
            *mat_params,
            # --- NEW: Pass strain limits to parent ---
            max_tensile_strain=max_tensile_strain,
            max_compressive_strain=max_compressive_strain
        )