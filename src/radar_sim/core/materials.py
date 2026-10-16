"""Surface materials and the two-component scattering model.

``a(psi) = rho * (1 - tau) * [ (1-eta) * exp(-psi^2 / (2*sigma_psi^2))
                              + eta * cos(psi) ] * 10^(-absorption_db/20)
           * exp(j * phase_deg)``

where ``psi`` is the incidence angle measured from the surface normal
(``0`` == facing the radar).  This is the engineering approximation agreed in
``docs/DESIGN.md`` section 4.3: a facing metal plate is very bright, a tilted
one dims rapidly, vegetation is weak but isotropic.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

DEG = math.pi / 180.0

__all__ = ["Material", "PRESETS", "preset_materials"]


@dataclass
class Material:
    id: str
    name: str = ""
    rho: float = 0.7            # reflectivity amplitude, 0..1
    eta: float = 0.3            # roughness, 0 = specular .. 1 = lambertian
    tau: float = 0.0            # transmittance, 0 = opaque .. 1 = clear
    absorption_db: float = 0.0  # extra loss per reflection
    phase_deg: float = 0.0      # constant phase offset on reflection
    sigma_psi_deg: float = 8.0  # specular lobe width
    penetration_db_per_m: float = 20.0
    color: str = "#9099a3"

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.id
        if not (0.0 <= self.rho <= 1.0):
            raise ValueError(f"rho must be within [0,1], got {self.rho}")
        if not (0.0 <= self.eta <= 1.0):
            raise ValueError(f"eta must be within [0,1], got {self.eta}")
        if not (0.0 <= self.tau <= 1.0):
            raise ValueError(f"tau must be within [0,1], got {self.tau}")

    # ------------------------------------------------------------------ #
    @property
    def opaque(self) -> bool:
        """``True`` when the material blocks the line of sight."""
        return self.tau < 0.5

    @property
    def transmittance_amplitude(self) -> float:
        return math.sqrt(max(self.tau, 0.0))

    def reflection_amplitude(self, psi_rad: np.ndarray | float) -> np.ndarray:
        """Complex reflection amplitude ``a(psi)``.

        ``psi`` is the incidence angle from the normal.  A zero normal
        (isotropic point target) should be passed ``psi = 0``.
        """
        psi = np.asarray(psi_rad, dtype=np.float64)
        sigma = max(self.sigma_psi_deg, 1e-3) * DEG
        spec = (1.0 - self.eta) * np.exp(-(psi ** 2) / (2.0 * sigma ** 2))
        diff = self.eta * np.cos(psi)
        amp = (
            self.rho
            * (1.0 - self.tau)
            * (spec + diff)
            * (10.0 ** (-self.absorption_db / 20.0))
        )
        return amp * np.exp(1j * self.phase_deg * DEG)

    def transmission_loss_amplitude(self, distance_m: float) -> float:
        """Amplitude factor after travelling ``distance_m`` inside the material."""
        if self.tau >= 0.99:
            return 1.0
        return self.transmittance_amplitude * (10.0 ** (-self.penetration_db_per_m * distance_m / 20.0))

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Material":
        valid = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in valid})


# --------------------------------------------------------------------------- #
# presets
# --------------------------------------------------------------------------- #
PRESETS: tuple[Material, ...] = (
    Material("metal", "金属板", rho=0.95, eta=0.05, tau=0.0, sigma_psi_deg=6.0, color="#b8c4d4"),
    Material("concrete", "混凝土墙", rho=0.45, eta=0.55, tau=0.0, sigma_psi_deg=12.0, color="#9a9a92"),
    Material("glass", "玻璃", rho=0.25, eta=0.10, tau=0.85, sigma_psi_deg=6.0, color="#a8d8e8"),
    Material("wood", "木板", rho=0.30, eta=0.70, tau=0.0, sigma_psi_deg=15.0, color="#c8a878"),
    Material("human", "人体", rho=0.35, eta=0.60, tau=0.0, sigma_psi_deg=14.0, color="#e8c0a0"),
    Material("vegetation", "植被", rho=0.12, eta=0.95, tau=0.0, sigma_psi_deg=20.0, color="#6f9f6f"),
    Material("absorber", "吸波材料", rho=0.02, eta=0.90, tau=0.0, sigma_psi_deg=20.0, color="#4a4a5a"),
    Material("custom", "自定义", rho=0.50, eta=0.50, tau=0.0, sigma_psi_deg=10.0, color="#c0c0c0"),
)


def preset_materials() -> dict[str, Material]:
    """Fresh ordered mapping ``id -> Material`` (dicts preserve order)."""
    return {m.id: Material.from_dict(m.to_dict()) for m in PRESETS}
