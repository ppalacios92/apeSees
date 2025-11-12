from __future__ import annotations
from typing import Any, Optional

# --- Use a relative import ---
from .base import Material

class Steel02_ape(Material):
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
    fu : float
        Ultimate strength.
    epsilon_u : float
        Strain at ultimate strength.
    r0 : float, optional
        Parameter controlling transition from elastic to plastic. (Default: 20.0)
    cr1 : float, optional
        Parameter controlling transition from elastic to plastic. (Default: 0.925)
    cr2 : float, optional
        Parameter controlling transition from elastic to plastic. (Default: 0.15)
    a1-a4 : float, optional
        Isotropic hardening parameters.
    max_tensile_strain : float, optional
        Tensile strain limit. If None, defaults to `epsilon_u`.
    max_compressive_strain : float, optional
        Compressive strain limit. If None, defaults to `-epsilon_u`.
    """
    def __init__(self,
                 tag: int,
                 fy: float,
                 E0: float,
                 fu:float,
                 epsilon_u:float,
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
        
        # --- Parameter calculation (unchanged) ---
        epsilon_y = fy / E0
        b = (fu - fy) / (E0 * (epsilon_u - epsilon_y))
        
        # Collect all parameters in the exact positional order
        mat_params = [fy, E0, b, r0, cr1, cr2]
        
        # Add optional isotropic hardening parameters if provided
        opt_params = [a1, a2, a3, a4]
        if any(p is not None for p in opt_params):
            a1 = a1 if a1 is not None else 0.0
            a2 = a2 if a2 is not None else 1.0
            a3 = a3 if a3 is not None else 0.0
            a4 = a4 if a4 is not None else 1.0
            mat_params.extend([a1, a2, a3, a4])

        # --- NEW: Set smart defaults for strain limits ---
        
        # 1. Tensile limit
        if max_tensile_strain is None:
            max_tensile_strain = epsilon_u

        # 2. Compressive limit (symmetric)
        if max_compressive_strain is None:
            max_compressive_strain = -epsilon_u
        # --- END NEW LOGIC ---

        # Call the parent class __init__
        super().__init__(
            "Steel02", 
            tag, 
            *mat_params,
            # --- NEW: Pass strain limits to parent ---
            max_tensile_strain=max_tensile_strain,
            max_compressive_strain=max_compressive_strain
        )