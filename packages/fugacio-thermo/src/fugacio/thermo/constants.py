"""Physical constants and unit conversions used across :mod:`fugacio.thermo`.

Fugacio works in SI internally:

* temperature ``T`` in kelvin (K),
* pressure ``P`` in pascal (Pa),
* amount in mole (mol),
* molar energy in joule per mole (J/mol),
* molar volume in cubic metre per mole (m^3/mol).

Molar mass is the one deliberate exception: it is stored in the conventional
gram-per-mole (g/mol), matching how component data are tabulated. Convert to
kg/mol with :data:`G_PER_MOL_TO_KG_PER_MOL` when you need mass in SI.

The numerical values follow the 2018 CODATA recommended values, so they match
the constants used by the open reference codes Fugacio is graded against.
"""

from __future__ import annotations

#: Universal gas constant ``R`` (J/mol/K), CODATA 2018.
R = 8.314462618

#: Avogadro constant ``N_A`` (1/mol), CODATA 2018.
N_A = 6.02214076e23

#: Boltzmann constant ``k_B`` (J/K), CODATA 2018.
K_B = 1.380649e-23

#: Reference temperature for thermochemistry (K), i.e. 25 degrees Celsius.
T_REF = 298.15

#: Standard-state pressure (Pa), 1 bar -- the IUPAC standard since 1982.
P_REF = 1.0e5

#: Zero of the Celsius scale, expressed in kelvin.
T_ZERO_CELSIUS = 273.15

# --- Pressure unit conversions (multiply a value in the named unit to get Pa) ---

#: 1 bar in pascal.
BAR = 1.0e5
#: 1 standard atmosphere in pascal.
ATM = 101325.0
#: 1 kilopascal in pascal.
KPA = 1.0e3
#: 1 millimetre of mercury (torr) in pascal.
MMHG = 133.32236842105263
#: 1 pound per square inch in pascal.
PSI = 6894.757293168361

#: Convert a molar mass in g/mol to kg/mol.
G_PER_MOL_TO_KG_PER_MOL = 1.0e-3


def celsius_to_kelvin(t_celsius: float) -> float:
    """Convert a temperature from degrees Celsius to kelvin."""
    return t_celsius + T_ZERO_CELSIUS


def kelvin_to_celsius(t_kelvin: float) -> float:
    """Convert a temperature from kelvin to degrees Celsius."""
    return t_kelvin - T_ZERO_CELSIUS
