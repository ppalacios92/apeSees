from __future__ import annotations
from typing import Optional

# --- Use a relative import ---
from .base import Material

class Hysteretic(Material):
    """
    Represents the OpenSees Hysteretic uniaxial material.
    
    This material model is defined by a multi-linear backbone and
    parameters controlling pinching, damage, and softening.
    
    Parameters
    ----------
    tag : int
        Unique material tag.
    s1p : float
        Stress at first positive backbone point.
    e1p : float
        Strain at first positive backbone point.
    s2p : float
        Stress at second positive backbone point.
    e2p : float
        Strain at second positive backbone point.
    s1n : float
        Stress at first negative backbone point (must be negative).
    e1n : float
        Strain at first negative backbone point (must be negative).
    s2n : float
        Stress at second negative backbone point (must be negative).
    e2n : float
        Strain at second negative backbone point (must be negative).
    pinch_x : float
        Pinching factor for strain (or deformation) during reloading.
    pinch_y : float
        Pinching factor for stress (or force) during reloading.
    damage1 : float
        Damage parameter for stiffness degradation.
    damage2 : float
        Damage parameter for strength degradation.
    s3p : float, optional
        Stress at third positive backbone point.
    e3p : float, optional
        Strain at third positive backbone point.
    s3n : float, optional
        Stress at third negative backbone point.
    e3n : float, optional
        Strain at third negative backbone point.
    beta : float, optional
        Power parameter for softening.
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
                 # --- KWARGS ---
                 max_tensile_strain: Optional[float] = None,
                 max_compressive_strain: Optional[float] = None):
        
        # --- Store all init parameters as attributes ---
        self.tag = tag
        self.s1p = s1p
        self.e1p = e1p
        self.s2p = s2p
        self.e2p = e2p
        self.s1n = s1n
        self.e1n = e1n
        self.s2n = s2n
        self.e2n = e2n
        self.s3p = s3p
        self.e3p = e3p
        self.s3n = s3n
        self.e3n = e3n
        self.pinch_x = pinch_x
        self.pinch_y = pinch_y
        self.damage1 = damage1
        self.damage2 = damage2
        self.beta = beta
        # Store the *original* input values, which might be None
        self.input_max_tensile_strain = max_tensile_strain
        self.input_max_compressive_strain = max_compressive_strain

        # --- Parameter list logic ---
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
            
        # --- Set smart defaults for strain limits ---
        
        # Start with the input values
        resolved_max_compressive_strain = max_compressive_strain
        resolved_max_tensile_strain = max_tensile_strain

        # 1. Compressive limit
        if resolved_max_compressive_strain is None:
            if is_trilinear and e3n is not None:
                resolved_max_compressive_strain = e3n # Use last point on envelope
            else:
                resolved_max_compressive_strain = e2n # Use 2nd point on envelope

        # 2. Tensile limit
        if resolved_max_tensile_strain is None:
            if is_trilinear and e3p is not None:
                resolved_max_tensile_strain = e3p # Use last point on envelope
            else:
                resolved_max_tensile_strain = e2p # Use 2nd point on envelope
        # --- END NEW LOGIC ---

        # Call the parent class __init__
        super().__init__(
            "Hysteretic", 
            tag, 
            *mat_params,
            # --- Pass *resolved* strain limits to parent ---
            max_tensile_strain=resolved_max_tensile_strain,
            max_compressive_strain=resolved_max_compressive_strain
        )