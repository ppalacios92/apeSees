from __future__ import annotations
from typing import Optional

# --- Use a relative import ---
from .base import Material

class Concrete01(Material):
    """
    Represents the OpenSees Concrete01 uniaxial material.

    This is the Kent-Scott-Park concrete model with degraded linear
    unloading/reloading stiffness and no tensile strength.

    OpenSees Documentation Parameters:
    matTag, fpc, epsc0, fpcu, epsU
    
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
    max_tensile_strain : float, optional
        Tensile strain limit. Default is 1e10 (no limit).
    max_compressive_strain : float, optional
        Compressive strain limit. If None, defaults to `epsu`.
    """
    def __init__(self,
                 tag: int,
                 fpc: float,
                 epsc0: float,
                 fpcu: float,
                 epsu: float,
                 *,
                 # --- KWARGS ---
                 max_tensile_strain: Optional[float] = None,
                 max_compressive_strain: Optional[float] = None):
        
        # --- Validation logic ---
        if fpc > 0:
            print(f"Warning: fpc ({fpc}) should be negative. Converting.")
            fpc = -fpc
        if epsc0 > 0:
            print(f"Warning: epsc0 ({epsc0}) should be negative. Converting.")
            epsc0 = -epsc0
        if fpcu > 0:
             print(f"Warning: fpcu ({fpcu}) is positive. It's usually zero or negative. Converting.")
             fpcu = -fpcu
        if epsu > 0:
            print(f"Warning: epsu ({epsu}) should be negative. Converting.")
            epsu = -epsu
        if fpcu > fpc:
            print(f"Warning: Crushing strength fpcu ({fpcu}) is greater than max strength fpc ({fpc}).")
        if epsu < epsc0:
            print(f"Warning: Crushing strain epsu ({epsu}) is less than strain at max strength epsc0 ({epsc0}).")

        # --- Store all init parameters as attributes ---
        self.tag = tag
        self.fpc = fpc
        self.epsc0 = epsc0
        self.fpcu = fpcu
        self.epsu = epsu
        # Store the *original* input values, which might be None
        self.input_max_tensile_strain = max_tensile_strain
        self.input_max_compressive_strain = max_compressive_strain

        # Collect all parameters in the exact positional order
        mat_params = [fpc, epsc0, fpcu, epsu]

        # --- Set smart default for compressive strain ---
        
        # Start with the input value
        resolved_max_compressive_strain = max_compressive_strain

        # If the user doesn't specify a limit, use the material's
        # ultimate strain (epsu) as the failure limit.
        if resolved_max_compressive_strain is None:
            resolved_max_compressive_strain = epsu  # epsu is already negative
        
        # Note: Concrete01 has no tension side, so we just pass the
        # user-provided max_tensile_strain (which is likely None)
        # to the parent, which will apply its own default (1e10).

        # Call the parent class __init__
        super().__init__(
            "Concrete01", 
            tag, 
            *mat_params,
            # --- Pass strain limits to parent ---
            max_tensile_strain=max_tensile_strain,
            max_compressive_strain=resolved_max_compressive_strain
        )