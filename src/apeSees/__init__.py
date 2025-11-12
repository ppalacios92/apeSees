# Removed the stray 'from re import A'

from .materials import Material, UniaxialMaterialTester, MaterialTestResult

# Imports from the section module
from .section import (
    Section, 
    # Add the new general classes
    GeneralFiberSection,
    PatchDefinition,
    FiberDefinition,
    # Keep the specific implementations
    RectangularColumnSection, 
    RectangularSolidSection,
    # Keep the analysis/result classes
    MomentCurvature,
    SectionResults,
    MomentCurvatureResults,
    FiberMapper
)

from .timeseries import (
    LinearTimeSeries, ConstantTimeSeries, PathTimeSeries, 
    ASCE41Protocol, ModifiedATC24Protocol, FEMA461Protocol
)
from .utilities import AttrDict

# __all__ now matches the imported classes
__all__ = [
    # Materials
    "Material",
    "UniaxialMaterialTester",
    "MaterialTestResult",
    
    # Sections
    "Section",
    "GeneralFiberSection",      # Added
    "PatchDefinition",          # Added
    "FiberDefinition",          # Added
    "RectangularColumnSection",
    "RectangularSolidSection",
    "MomentCurvature",
    "SectionResults",
    "MomentCurvatureResults",
    "FiberMapper",
    
    # Time Series
    "LinearTimeSeries",
    "ConstantTimeSeries",
    "PathTimeSeries",
    "ASCE41Protocol",
    "ModifiedATC24Protocol",
    "FEMA461Protocol",
    
    # Utilities
    "AttrDict",
]