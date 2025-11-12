from __future__ import annotations
from typing import Optional

# --- Use a relative import ---
from .base import Material

class Hysteretic(Material):
    """
    Represents the OpenSees Hysteretic uniaxial material.
    ... (docstring unchanged) ...
    
    Parameters
    ----------
    ... (standard parameters unchanged) ...
    
    max_tensile_strain : float, optional
        Tensile strain limit. If None, defaults to `e3p` if provided,
        otherwise `e2p`.
    max_compressive_strain : float, optional
        Compressive strain limit. If None, defaults to `e3n` if provided,
        otherwise `e2n`.
    """
    def __init__(self,
                 tag: int,
                 s1p: float, e1p: float,
                 s2p: float, e2p: float,
                 s1n: float, e1n: float,
                 s2n: float, e2n: float,
                 pinch_x: float,
                 pinch_y: float,
                 damage1: float,
                 damage2: float,
                 s3p: Optional[float] = None,
                 e3p: Optional[float] = None,
                 s3n: Optional[float] = None,
                 e3n: Optional[float] = None,
                 beta: Optional[float] = None,
                 *,
                 # --- NEW KWARGS ---
                 max_tensile_strain: Optional[float] = None,
                 max_compressive_strain: Optional[float] = None):
        
        # --- Parameter list logic (unchanged) ---
        mat_params = [s1p, e1p, s2p, e2p]

        is_trilinear = False
        if (s3p is not None) and (e3p is not None):
            mat_params.extend([s3p, e3p])
            is_trilinear = True # We have a 3rd positive point
        elif (s3p is not None) or (e3p is not None):
            raise ValueError("Both 's3p' and 'e3p' must be provided together, or both omitted.")

        mat_params.extend([s1n, e1n, s2n, e2n])

        if (s3n is not None) and (e3n is not None):
            mat_params.extend([s3n, e3n])
            # We assume if s3p/e3p are set, s3n/e3n are too
        elif (s3n is not None) or (e3n is not None):
            raise ValueError("Both 's3n' and 'e3n' must be provided together, or both omitted.")

        mat_params.extend([pinch_x, pinch_y, damage1, damage2])
        
        if beta is not None:
            mat_params.append(beta)
            
        # --- NEW: Set smart defaults for strain limits ---
        
        # 1. Compressive limit
        if max_compressive_strain is None:
            if is_trilinear and e3n is not None:
                max_compressive_strain = e3n # Use last point on envelope
            else:
                max_compressive_strain = e2n # Use 2nd point on envelope

        # 2. Tensile limit
        if max_tensile_strain is None:
            if is_trilinear and e3p is not None:
                max_tensile_strain = e3p # Use last point on envelope
            else:
                max_tensile_strain = e2p # Use 2nd point on envelope
        # --- END NEW LOGIC ---

        # Call the parent class __init__
        super().__init__(
            "Hysteretic", 
            tag, 
            *mat_params,
            # --- NEW: Pass strain limits to parent ---
            max_tensile_strain=max_tensile_strain,
            max_compressive_strain=max_compressive_strain
        )