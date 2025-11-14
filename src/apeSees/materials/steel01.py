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
                 # --- KWARGS ---
                 max_tensile_strain: Optional[float] = None,
                 max_compressive_strain: Optional[float] = None):
        
        # --- Store all init parameters as attributes ---
        self.tag = tag
        self.fy = fy
        self.e = e
        self.b = b
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        # Store the *original* input values, which might be None
        self.input_max_tensile_strain = max_tensile_strain
        self.input_max_compressive_strain = max_compressive_strain

        # --- Parameter list logic ---
        mat_params = [fy, e, b]
        
        opt_params = [a1, a2, a3, a4]
        
        # We need to *use* the local variables a1, a2, etc.
        # for the extension logic, not self.a1
        if any(p is not None for p in opt_params):
            a1_resolved = a1 if a1 is not None else 0.0
            a2_resolved = a2 if a2 is not None else 1.0
            a3_resolved = a3 if a3 is not None else 0.0
            a4_resolved = a4 if a4 is not None else 1.0
            mat_params.extend([a1_resolved, a2_resolved, a3_resolved, a4_resolved])

        # --- Call the parent class __init__ ---
        super().__init__(
            "Steel01", 
            tag, 
            *mat_params,
            # --- Pass strain limits to parent ---
            # No smart defaults needed for Steel01
            max_tensile_strain=max_tensile_strain,
            max_compressive_strain=max_compressive_strain
        )