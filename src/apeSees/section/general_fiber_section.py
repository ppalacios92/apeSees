from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Set
from dataclasses import dataclass
import openseespy.opensees as ops
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle

from .base import Section
from ..materials import Material

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

# --- 1. Define the building blocks ---

@dataclass
class FiberDefinition:
    """A single fiber."""
    y: float
    z: float
    area: float
    material: Material

@dataclass
class PatchDefinition:
    """A patch of fibers (rectangular or polygonal)."""
    material: Material
    num_fibers_y: int  # Number of fibers along y
    num_fibers_z: int  # Number of fibers along z
    coords: List[float]  # [yI, zI, yJ, zJ] for 'rect'
                         # [y1, z1, y2, z2, ...] for 'poly'
    type: str = 'rect'   # 'rect' or 'poly'
    

# --- 2. Create the General Section class ---

class GeneralFiberSection(Section):
    """
    A general fiber section defined by a list of patches and fibers.
    This class handles the OpenSees 'build' and 'plot_section' logic.
    """
    
    def __init__(
        self,
        section_tag: int,
        patches: List[PatchDefinition],
        fibers: List[FiberDefinition] = [],
        GJ: Optional[float] = None
    ):
        self.section_tag = int(section_tag)
        self.patches = patches
        self.fibers = fibers
        self.GJ = GJ
        
    def build(self, verbose: bool = False) -> int:
        """
        Build the section in OpenSees by looping over patches and fibers.
        """
        if self.GJ is not None:
            ops.section('Fiber', self.section_tag, '-GJ', self.GJ)
        else:
            ops.section('Fiber', self.section_tag)

        if verbose:
            print(f"Building GeneralFiberSection (tag={self.section_tag})")
            if self.GJ:
                print(f"  GJ: {self.GJ:.2e} N·mm²")

        # --- THIS IS THE FIX ---
        # Get all materials from our new method
        all_materials = self.get_materials()
        # --- END FIX ---
        
        if verbose and all_materials:
            print(f"  Building {len(all_materials)} unique materials...")
        for mat in all_materials:
            mat.build() 

        # Build patches
        for i, patch in enumerate(self.patches):
            if patch.type == 'rect':
                ops.patch('rect', patch.material.tag, 
                          patch.num_fibers_y, patch.num_fibers_z, 
                          *patch.coords)
                if verbose:
                    print(f"  Patch {i} (rect): {patch.num_fibers_y}×{patch.num_fibers_z} fibers")
            elif patch.type == 'poly':
                ops.patch('poly', patch.material.tag, 
                          patch.num_fibers_y, patch.num_fibers_z, 
                          *patch.coords)
                if verbose:
                    print(f"  Patch {i} (poly): {patch.num_fibers_y}×{patch.num_fibers_z} fibers")
            # Add 'circ' etc. as needed

        # Build individual fibers (e.g., for rebars)
        for i, fiber in enumerate(self.fibers):
            ops.fiber(fiber.y, fiber.z, fiber.area, fiber.material.tag)
        
        if verbose and self.fibers:
            print(f"  Fibers: {len(self.fibers)} individual fibers")
            
        return self.section_tag

    # --- NEW METHOD (REQUIRED BY BASE CLASS) ---
    def get_materials(self) -> Set[Material]:
        """
        Return a set of all unique Material objects used in the section.
        """
        all_materials = set()
        for patch in self.patches:
            all_materials.add(patch.material)
        for fiber in self.fibers:
            all_materials.add(fiber.material)
        return all_materials
    # --- END NEW METHOD ---

    def plot_section(
        self, 
        ax: Optional[plt.Axes] = None, 
        figsize: tuple[float, float] = (6, 6),
        patch_color: str = '#d9d9d9',
        fiber_color: str = 'black',
        **kwargs
    ) -> plt.Axes:
        """
        Plot the section geometry by looping over patches and fibers.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        all_y = []
        all_z = []

        # Plot patches
        for patch in self.patches:
            if patch.type == 'rect':
                yI, zI, yJ, zJ = patch.coords
                width = yJ - yI
                height = zJ - zI
                ax.add_patch(Rectangle((yI, zI), width, height, 
                                       facecolor=patch_color, edgecolor='black', lw=0.5))
                all_y.extend([yI, yJ])
                all_z.extend([zI, zJ])
            # Add logic for 'poly' using Polygon patch

        # Plot fibers
        for fiber in self.fibers:
            radius = (fiber.area / 3.14159)**0.5 
            ax.add_patch(Circle((fiber.y, fiber.z), radius, 
                                color=fiber_color, zorder=10))
            all_y.extend([fiber.y - radius, fiber.y + radius])
            all_z.extend([fiber.z - radius, fiber.z + radius])

        ax.set_aspect('equal')
        ax.set_xlabel("y")
        ax.set_ylabel("z")
        ax.set_title(f"Section (tag={self.section_tag})")
        ax.grid(True, alpha=0.3)
        
        # Auto-set limits
        if all_y and all_z:
            min_y, max_y = min(all_y), max(all_y)
            min_z, max_z = min(all_z), max(all_z)
            pad_y = (max_y - min_y) * 0.05 + 1.0
            pad_z = (max_z - min_z) * 0.05 + 1.0
            ax.set_xlim(min_y - pad_y, max_y + pad_y)
            ax.set_ylim(min_z - pad_z, max_z + pad_z)

        return ax

    def __repr__(self) -> str:
        return (
            f"GeneralFiberSection(tag={self.section_tag}, "
            f"patches={len(self.patches)}, fibers={len(self.fibers)})"
        )