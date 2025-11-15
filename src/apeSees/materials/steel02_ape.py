from __future__ import annotations
from typing import Any, Optional

# --- Use a relative import ---
from .base import Material

class Steel02_ape(Material):
    """
    Represents the OpenSees Steel02 uniaxial material (Giuffré-Menegotto-Pinto).

    This class inherits from the base Material class and provides
    a specific interface for the Steel02 parameters, deriving the
    strain-hardening ratio `b` from `fu` and `epsilon_u`.

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
                 # --- KWARGS ---
                 max_tensile_strain: Optional[float] = None,
                 max_compressive_strain: Optional[float] = None):
        
        # --- Store all init parameters as attributes ---
        self.tag = tag
        self.fy = fy
        self.E0 = E0
        self.fu = fu
        self.epsilon_u = epsilon_u
        self.r0 = r0
        self.cr1 = cr1
        self.cr2 = cr2
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        # Store the *original* input values, which might be None
        self.input_max_tensile_strain = max_tensile_strain
        self.input_max_compressive_strain = max_compressive_strain

        # --- Parameter calculation ---
        epsilon_y = fy / E0
        # Handle potential zero-strain hardening
        if abs(epsilon_u - epsilon_y) < 1e-10:
            b = 0.0 # No hardening range
        else:
            b = (fu - fy) / (E0 * (epsilon_u - epsilon_y))
            # Ensure b is not negative
            if b < 0.0:
                print(f"Warning: Calculated hardening ratio 'b' ({b}) is negative. Setting to 0.0.")
                b = 0.0
        
        self.b = b # Store the calculated hardening ratio
        
        # Collect all parameters in the exact positional order
        mat_params = [fy, E0, b, r0, cr1, cr2]
        
        # Add optional isotropic hardening parameters if provided
        opt_params = [a1, a2, a3, a4]
        if any(p is not None for p in opt_params):
            a1_resolved = a1 if a1 is not None else 0.0
            a2_resolved = a2 if a2 is not None else 1.0
            a3_resolved = a3 if a3 is not None else 0.0
            a4_resolved = a4 if a4 is not None else 1.0
            mat_params.extend([a1_resolved, a2_resolved, a3_resolved, a4_resolved])

        # --- Set smart defaults for strain limits ---
        
        # Start with the input values
        resolved_max_tensile_strain = max_tensile_strain
        resolved_max_compressive_strain = max_compressive_strain

        # 1. Tensile limit
        if resolved_max_tensile_strain is None:
            resolved_max_tensile_strain = epsilon_u

        # 2. Compressive limit (symmetric)
        if resolved_max_compressive_strain is None:
            # Ensure it's negative
            resolved_max_compressive_strain = -abs(epsilon_u)
        # --- END NEW LOGIC ---

        # Call the parent class __init__
        super().__init__(
            "Steel02", 
            tag, 
            *mat_params,
            # --- Pass *resolved* strain limits to parent ---
            max_tensile_strain=resolved_max_tensile_strain,
            max_compressive_strain=resolved_max_compressive_strain
        )