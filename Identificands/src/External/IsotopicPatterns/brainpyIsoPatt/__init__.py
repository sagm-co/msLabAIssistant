"""
A Python Implementation of the
Baffling Recursive Algorithm for Isotopic cluster distributioN
"""
import os

from .brainpyIsoPatt import (isotopic_variants, IsotopicDistribution,
                      max_variants, calculate_mass, neutral_mass, mass_charge_ratio, Peak, max_variants_approx)

from .composition import parse_formula, PyComposition


SimpleComposition = PyComposition


