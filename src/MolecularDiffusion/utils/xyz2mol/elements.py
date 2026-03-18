"""Element data for XYZ-to-mol conversion.

Trimmed from the original ElementData class to include only the properties
used by the converter and featurizer modules:
  elementnr, elementsym, valenceelectrons, elementperiod,
  elementgroup, elementblock, ElectroNegativityPauling
"""


class ElementData:

    def __init__(self):
        self.elementnr = {
            "Em": 0, "Vc": 0, "Va": 0, "D": 1,
            "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6,
            "N": 7, "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12,
            "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
            "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24,
            "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
            "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
            "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,
            "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48,
            "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54,
            "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
            "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66,
            "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72,
            "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
            "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84,
            "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
            "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96,
            "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100, "Md": 101, "No": 102,
            "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107,
            "Hs": 108, "Mt": 109, "Uun": 110, "Uuu": 111, "Uub": 112,
        }

        self.elementsym = {v: k for k, v in self.elementnr.items()}

        self.valenceelectrons = {
            "Em": 0, "Vc": 0, "Va": 0, "H": 1, "D": 1, "He": 0,
            "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7,
            "Ne": 0, "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": 5, "S": 6,
            "Cl": 7, "Ar": 0, "K": 1, "Ca": 2, "Sc": 3, "Ti": 4, "V": 5,
            "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10, "Cu": 1,
            "Zn": 2, "Ga": 3, "Ge": 4, "As": 5, "Se": 6, "Br": 7,
            "Kr": 0, "Rb": 1, "Sr": 2, "Y": 3, "Zr": 4, "Nb": 5,
            "Mo": 6, "Tc": 7, "Ru": 8, "Rh": 9, "Pd": 10, "Ag": 1,
            "Cd": 2, "In": 3, "Sn": 4, "Sb": 5, "Te": 6, "I": 7,
            "Xe": 0, "Cs": 1, "Ba": 2, "La": 3, "Ce": 2, "Pr": 2,
            "Nd": 2, "Pm": 2, "Sm": 2, "Eu": 2, "Gd": 2, "Tb": 2,
            "Dy": 2, "Ho": 2, "Er": 2, "Tm": 2, "Yb": 2, "Lu": 3,
            "Hf": 4, "Ta": 5, "W": 6, "Re": 7, "Os": 8, "Ir": 9,
            "Pt": 10, "Au": 1, "Hg": 2, "Tl": 3, "Pb": 4, "Bi": 5,
            "Po": 6, "At": 7, "Rn": 0, "Fr": 1, "Ra": 2, "Ac": 3,
            "Th": 4, "Pa": 5, "U": 6, "Np": 7, "Pu": 8, "Am": 2,
            "Cm": 2, "Bk": 2, "Cf": 2, "Es": 2,
        }

        self.elementperiod = {
            "H": 1, "D": 1, "He": 2,
            "Li": 2, "Be": 2, "B": 2, "C": 2, "N": 2, "O": 2, "F": 2, "Ne": 2,
            "Na": 3, "Mg": 3, "Al": 3, "Si": 3, "P": 3, "S": 3, "Cl": 3, "Ar": 3,
            "K": 4, "Ca": 4, "Sc": 4, "Ti": 4, "V": 4, "Cr": 4, "Mn": 4,
            "Fe": 4, "Co": 4, "Ni": 4, "Cu": 4, "Zn": 4, "Ga": 4, "Ge": 4,
            "As": 4, "Se": 4, "Br": 4, "Kr": 4,
            "Rb": 5, "Sr": 5, "Y": 5, "Zr": 5, "Nb": 5, "Mo": 5, "Tc": 5,
            "Ru": 5, "Rh": 5, "Pd": 5, "Ag": 5, "Cd": 5, "In": 5, "Sn": 5,
            "Sb": 5, "Te": 5, "I": 5, "Xe": 5,
            "Cs": 6, "Ba": 6, "La": 6, "Ce": 6, "Pr": 6, "Nd": 6, "Pm": 6,
            "Sm": 6, "Eu": 6, "Gd": 6, "Tb": 6, "Dy": 6, "Ho": 6, "Er": 6,
            "Tm": 6, "Yb": 6, "Lu": 6, "Hf": 6, "Ta": 6, "W": 6, "Re": 6,
            "Os": 6, "Ir": 6, "Pt": 6, "Au": 6, "Hg": 6, "Tl": 6, "Pb": 6,
            "Bi": 6, "Po": 6, "At": 6, "Rn": 6,
            "Fr": 7, "Ra": 7, "Ac": 7, "Th": 7, "Pa": 7, "U": 7, "Np": 7,
            "Pu": 7, "Am": 7, "Cm": 7, "Bk": 7, "Cf": 7, "Es": 7, "Fm": 7,
            "Md": 7, "No": 7, "Lr": 7, "Rf": 7, "Db": 7, "Sg": 7, "Bh": 7,
            "Hs": 7, "Mt": 7, "Uun": 7, "Uuu": 7, "Uub": 7,
        }

        self.elementgroup = {
            "H": 1, "D": 1, "He": 18,
            "Li": 1, "Be": 2, "B": 13, "C": 14, "N": 15, "O": 16, "F": 17, "Ne": 18,
            "Na": 1, "Mg": 2, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
            "K": 1, "Ca": 2, "Sc": 3, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7,
            "Fe": 8, "Co": 9, "Ni": 10, "Cu": 11, "Zn": 12, "Ga": 13, "Ge": 14,
            "As": 15, "Se": 16, "Br": 17, "Kr": 18,
            "Rb": 1, "Sr": 2, "Y": 3, "Zr": 4, "Nb": 5, "Mo": 6, "Tc": 7,
            "Ru": 8, "Rh": 9, "Pd": 10, "Ag": 11, "Cd": 12, "In": 13, "Sn": 14,
            "Sb": 15, "Te": 16, "I": 17, "Xe": 18,
            "Cs": 1, "Ba": 2,
            "La": 19, "Ce": 19, "Pr": 19, "Nd": 19, "Pm": 19, "Sm": 19,
            "Eu": 19, "Gd": 19, "Tb": 19, "Dy": 19, "Ho": 19, "Er": 19,
            "Tm": 19, "Yb": 19,
            "Lu": 3, "Hf": 4, "Ta": 5, "W": 6, "Re": 7, "Os": 8, "Ir": 9,
            "Pt": 10, "Au": 11, "Hg": 12, "Tl": 13, "Pb": 14, "Bi": 15,
            "Po": 16, "At": 17, "Rn": 18,
            "Fr": 1, "Ra": 2,
            "Ac": 19, "Th": 19, "Pa": 19, "U": 19, "Np": 19, "Pu": 19,
            "Am": 19, "Cm": 19, "Bk": 19, "Cf": 19, "Es": 19, "Fm": 19,
            "Md": 19, "No": 19,
            "Lr": 3, "Rf": 4, "Db": 5, "Sg": 6, "Bh": 7, "Hs": 8, "Mt": 9,
            "Uun": 10, "Uuu": 11, "Uub": 12,
        }

        self.elementblock = {
            "Em": "s", "Vc": "s", "Va": "s", "H": "s", "D": "s", "He": "s",
            "Li": "s", "Be": "s",
            "B": "p", "C": "p", "N": "p", "O": "p", "F": "p", "Ne": "p",
            "Na": "s", "Mg": "s",
            "Al": "p", "Si": "p", "P": "p", "S": "p", "Cl": "p", "Ar": "p",
            "K": "s", "Ca": "s",
            "Sc": "d", "Ti": "d", "V": "d", "Cr": "d", "Mn": "d", "Fe": "d",
            "Co": "d", "Ni": "d", "Cu": "d", "Zn": "d",
            "Ga": "p", "Ge": "p", "As": "p", "Se": "p", "Br": "p", "Kr": "p",
            "Rb": "s", "Sr": "s",
            "Y": "d", "Zr": "d", "Nb": "d", "Mo": "d", "Tc": "d", "Ru": "d",
            "Rh": "d", "Pd": "d", "Ag": "d", "Cd": "d",
            "In": "p", "Sn": "p", "Sb": "p", "Te": "p", "I": "p", "Xe": "p",
            "Cs": "s", "Ba": "s",
            "La": "f", "Ce": "f", "Pr": "f", "Nd": "f", "Pm": "f", "Sm": "f",
            "Eu": "f", "Gd": "f", "Tb": "f", "Dy": "f", "Ho": "f", "Er": "f",
            "Tm": "f", "Yb": "f", "Lu": "f",
            "Hf": "d", "Ta": "d", "W": "d", "Re": "d", "Os": "d", "Ir": "d",
            "Pt": "d", "Au": "d", "Hg": "d",
            "Tl": "p", "Pb": "p", "Bi": "p", "Po": "p", "At": "p", "Rn": "p",
            "Fr": "s", "Ra": "s",
            "Ac": "f", "Th": "f", "Pa": "f", "U": "f", "Np": "f", "Pu": "f",
            "Am": "f", "Cm": "f", "Bk": "f", "Cf": "f", "Es": "f", "Fm": "f",
            "Md": "f", "No": "f",
            "Lr": "d", "Rf": "d", "Db": "d", "Sg": "d", "Bh": "d", "Hs": "d",
            "Mt": "d", "Uun": "d", "Uuu": "d", "Uub": "d",
        }

        self.ElectroNegativityPauling = {
            "H": 2.20, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55,
            "N": 3.04, "O": 3.44, "F": 3.98, "Na": 0.93, "Mg": 1.31,
            "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16,
            "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63,
            "Cr": 1.66, "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91,
            "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01, "As": 2.18,
            "Se": 2.55, "Br": 2.96, "Kr": 3.00, "Rb": 0.82, "Sr": 0.95,
            "Y": 1.22, "Zr": 1.33, "Nb": 1.60, "Mo": 2.16, "Tc": 1.9,
            "Ru": 2.2, "Rh": 2.28, "Pd": 2.20, "Ag": 1.93, "Cd": 1.69,
            "In": 1.78, "Sn": 1.96, "Sb": 2.05, "Te": 2.1, "I": 2.66,
            "Xe": 2.6, "Cs": 0.79, "Ba": 0.89, "La": 1.10, "Ce": 1.12,
            "Pr": 1.13, "Nd": 1.14, "Sm": 1.17, "Gd": 1.20, "Dy": 1.22,
            "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Lu": 1.27, "Hf": 1.3,
            "Ta": 1.5, "W": 2.36, "Re": 1.9, "Os": 2.2, "Ir": 2.20,
            "Pt": 2.28, "Au": 2.54, "Hg": 2.00, "Tl": 1.62, "Pb": 2.33,
            "Bi": 2.02, "Po": 2.0, "At": 2.2, "Fr": 0.7, "Ra": 0.9,
            "Ac": 1.100, "Th": 1.300, "Pa": 1.500, "U": 1.380, "Np": 1.360,
            "Pu": 1.280, "Am": 1.300, "Cm": 1.300, "Bk": 1.300, "Cf": 1.300,
            "Es": 1.300, "Fm": 1.300, "Md": 1.300, "No": 1.300,
        }
