from __future__ import annotations
from typing import Any, Optional

# --- Use a relative import ---
from .base import Material

class Steel02(Material):
    """
    Represents the OpenSees Steel02 uniaxial material (Giuffré-Menegotto-Pinto).

    This class inherits from the base Material class and provides
    a specific interface for the Steel02 parameters.

    OpenSees Documentation Parameters:
    matTag, fy, E, b, R0, cR1, cR2, <a1, a2, a3, a4>
    
    Parameters
    ----------
    tag : int
        Unique material tag.
    fy : float
        Yield strength.
    E0 : float
        Young's modulus.
    b : float
        Strain hardening ratio (E_sh / E).
    r0 : float, optional
        Parameter controlling transition from elastic to plastic. (Default: 20.0)
    cr1 : float, optional
        Parameter controlling transition from elastic to plastic. (Default: 0.925)
    cr2 : float, optional
        Parameter controlling transition from elastic to plastic. (Default: 0.15)
    a1-a4 : float, optional
        Isotropic hardening parameters.
    max_tensile_strain : float, optional
        Tensile strain limit. Default is 1e10 (no limit).
    max_compressive_strain : float, optional
        Compressive strain limit. Default is -1e10 (no limit).
    """
    def __init__(self,
                 tag: int,
                 fy: float,
                 E0: float,
                 b: float,
                 r0: float = 20.0,
                 cr1: float = 0.925,
                 cr2: float = 0.15,
                 a1: Optional[float] = None,
                 a2: Optional[float] = None,
                 a3: Optional[float] = None,
                 a4: Optional[float] = None,
                 *,
                 # --- NEW KWARGS ---
                 max_tensile_strain: Optional[float] = None,
                 max_compressive_strain: Optional[float] = None):
        
        # --- Parameter list logic (unchanged) ---
        mat_params = [fy, E0, b, r0, cr1, cr2]
        
        opt_params = [a1, a2, a3, a4]
        if any(p is not None for p in opt_params):
            a1 = a1 if a1 is not None else 0.0
            a2 = a2 if a2 is not None else 1.0
            a3 = a3 if a3 is not None else 0.0
            a4 = a4 if a4 is not None else 1.0
            mat_params.extend([a1, a2, a3, a4])

        # --- Call the parent class __init__ (MODIFIED) ---
        super().__init__(
            "Steel02", 
            tag, 
            *mat_params,
            # --- NEW: Pass strain limits to parent ---
            max_tensile_strain=max_tensile_strain,
            max_compressive_strain=max_compressive_strain
        )