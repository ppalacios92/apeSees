from __future__ import annotations
from typing import Optional

# --- Use a relative import ---
from .base import Material

class Concrete02(Material):
    """
    Represents the OpenSees Concrete02 uniaxial material.

    This is a concrete model with linear tension softening.

    OpenSees Documentation Parameters:
    matTag, fpc, epsc0, fpcu, epsU, lambda, ft, Ets
    
    Parameters
    ----------
    tag : int
        Unique material tag.
    fpc : float
        Concrete compressive strength (must be a negative value).
    epsc0 : float
        Concrete strain at maximum strength (must be a negative value).
    fpcu : float
        Concrete crushing strength (must be a negative value).
    epsu : float
        Concrete strain at crushing strength (must be a negative value).
    lambda_val : float
        Ratio between unloading slope at epsU and initial slope (positive).
    ft : float
        Tensile strength (must be a positive value).
    Ets : float
        Tension softening stiffness (must be a positive value).
    max_tensile_strain : float, optional
        Tensile strain limit. If None, defaults to the calculated ultimate
        tensile strain: (ft/Ec) + (ft/Ets).
    max_compressive_strain : float, optional
        Compressive strain limit. If None, defaults to `epsu`.
    """
    def __init__(self,
                 tag: int,
                 fpc: float,
                 epsc0: float,
                 fpcu: float,
                 epsu: float,
                 lambda_val: float,
                 ft: float,
                 Ets: float,
                 *,
                 # --- NEW KWARGS ---
                 max_tensile_strain: Optional[float] = None,
                 max_compressive_strain: Optional[float] = None):
        
        # --- Validation logic (unchanged) ---
        if fpc > 0: fpc = -fpc
        if epsc0 > 0: epsc0 = -epsc0
        if fpcu > 0: fpcu = -fpcu
        if epsu > 0: epsu = -epsu
        if lambda_val < 0: lambda_val = abs(lambda_val)
        if ft < 0: ft = abs(ft)
        if Ets < 0: Ets = abs(Ets)
        # (Removed print warnings for brevity, you can keep them)

        # Collect all parameters in the exact positional order
        mat_params = [fpc, epsc0, fpcu, epsu, lambda_val, ft, Ets]

        # --- NEW: Set smart defaults for strain limits ---
        
        # 1. Compressive limit
        if max_compressive_strain is None:
            max_compressive_strain = epsu  # epsu is already negative

        # 2. Tensile limit
        if max_tensile_strain is None:
            if Ets > 1e-10: # Avoid division by zero if no tension softening
                # Initial elastic modulus
                Ec = (2 * fpc) / epsc0  # (negative / negative) = positive
                if Ec > 1e-10:
                    # Strain at peak tensile stress
                    epst_peak = ft / Ec
                    # Strain from peak to zero stress
                    epst_softening = ft / Ets
                    # Ultimate tensile strain
                    max_tensile_strain = epst_peak + epst_softening
                else:
                    max_tensile_strain = 1e10 # No stiffness, use default
            else:
                 max_tensile_strain = 1e10 # No softening, use default
        # --- END NEW LOGIC ---

        # Call the parent class __init__
        super().__init__(
            "Concrete02", 
            tag, 
            *mat_params,
            # --- NEW: Pass strain limits to parent ---
            max_tensile_strain=max_tensile_strain,
            max_compressive_strain=max_compressive_strain
        )