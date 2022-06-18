
def get_problem(name, *args, **kwargs):
    name = name.lower()

    if name.startswith("bbob-"):
        from pymoo.vendor.vendor_coco import COCOProblem
        return COCOProblem(name.lower(), **kwargs)

    from pymoo.problems.multi import BNH, Carside
    from pymoo.problems.multi import CTP1, CTP2, CTP3, CTP4, CTP5, CTP6, CTP7, CTP8
    from pymoo.problems.multi import DASCMOP1, DASCMOP2, DASCMOP3, DASCMOP4, DASCMOP5, DASCMOP6, DASCMOP7, DASCMOP8, \
        DASCMOP9
    from pymoo.problems.multi import MODAct, MW1, MW2, MW3, MW4, MW5, MW6, MW7, MW8, MW9, MW10, MW11, MW12, MW13, MW14
    from pymoo.problems.single import Ackley
    from pymoo.problems.many import DTLZ1, C1DTLZ1, DC1DTLZ1, DC1DTLZ3, DC2DTLZ1, DC2DTLZ3, DC3DTLZ1, DC3DTLZ3, C1DTLZ3, \
        C2DTLZ2, C3DTLZ1, C3DTLZ4, ScaledDTLZ1, ConvexDTLZ2, ConvexDTLZ4, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7, \
        InvertedDTLZ1, WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9
    from pymoo.problems.multi import Kursawe, OSY, SRN, TNK, Truss2D, WeldedBeam, ZDT1, ZDT2, ZDT3, ZDT4, ZDT5, ZDT6
    from pymoo.problems.single import CantileveredBeam, Griewank, Himmelblau, Knapsack, PressureVessel, Rastrigin, \
        Rosenbrock, Schwefel, Sphere, Zakharov
    from pymoo.problems.single import G1, G2, G3, G4, G5, G6, G7, G8, G9, G10, G11, G12, G13, G14, G15, G16, G17, G18, \
        G19, G20, G21, G22, G23, G24
    from pymoo.problems.dynamic.df import DF1, DF2, DF3, DF4, DF5, DF6, DF7, DF8, DF9, DF10, DF11, DF12, DF13, DF14

    PROBLEM = {
        'ackley': Ackley,
        'bnh': BNH,
        'carside': Carside,
        'ctp1': CTP1,
        'ctp2': CTP2,
        'ctp3': CTP3,
        'ctp4': CTP4,
        'ctp5': CTP5,
        'ctp6': CTP6,
        'ctp7': CTP7,
        'ctp8': CTP8,
        'dascmop1': DASCMOP1,
        'dascmop2': DASCMOP2,
        'dascmop3': DASCMOP3,
        'dascmop4': DASCMOP4,
        'dascmop5': DASCMOP5,
        'dascmop6': DASCMOP6,
        'dascmop7': DASCMOP7,
        'dascmop8': DASCMOP8,
        'dascmop9': DASCMOP9,
        'df1': DF1,
        'df2': DF2,
        'df3': DF3,
        'df4': DF4,
        'df5': DF5,
        'df6': DF6,
        'df7': DF7,
        'df8': DF8,
        'df9': DF9,
        'df10': DF10,
        'df11': DF11,
        'df12': DF12,
        'df13': DF13,
        'df14': DF14,
        'modact': MODAct,
        'mw1': MW1,
        'mw2': MW2,
        'mw3': MW3,
        'mw4': MW4,
        'mw5': MW5,
        'mw6': MW6,
        'mw7': MW7,
        'mw8': MW8,
        'mw9': MW9,
        'mw10': MW10,
        'mw11': MW11,
        'mw12': MW12,
        'mw13': MW13,
        'mw14': MW14,
        'dtlz1^-1': InvertedDTLZ1,
        'dtlz1': DTLZ1,
        'dtlz2': DTLZ2,
        'dtlz3': DTLZ3,
        'dtlz4': DTLZ4,
        'dtlz5': DTLZ5,
        'dtlz6': DTLZ6,
        'dtlz7': DTLZ7,
        'convex_dtlz2': ConvexDTLZ2,
        'convex_dtlz4': ConvexDTLZ4,
        'sdtlz1': ScaledDTLZ1,
        'c1dtlz1': C1DTLZ1,
        'c1dtlz3': C1DTLZ3,
        'c2dtlz2': C2DTLZ2,
        'c3dtlz1': C3DTLZ1,
        'c3dtlz4': C3DTLZ4,
        'dc1dtlz1': DC1DTLZ1,
        'dc1dtlz3': DC1DTLZ3,
        'dc2dtlz1': DC2DTLZ1,
        'dc2dtlz3': DC2DTLZ3,
        'dc3dtlz1': DC3DTLZ1,
        'dc3dtlz3': DC3DTLZ3,
        'cantilevered_beam': CantileveredBeam,
        'griewank': Griewank,
        'himmelblau': Himmelblau,
        'knp': Knapsack,
        'kursawe': Kursawe,
        'osy': OSY,
        'pressure_vessel': PressureVessel,
        'rastrigin': Rastrigin,
        'rosenbrock': Rosenbrock,
        'schwefel': Schwefel,
        'sphere': Sphere,
        'srn': SRN,
        'tnk': TNK,
        'truss2d': Truss2D,
        'welded_beam': WeldedBeam,
        'zakharov': Zakharov,
        'zdt1': ZDT1,
        'zdt2': ZDT2,
        'zdt3': ZDT3,
        'zdt4': ZDT4,
        'zdt5': ZDT5,
        'zdt6': ZDT6,
        'g1': G1,
        'g2': G2,
        'g3': G3,
        'g4': G4,
        'g5': G5,
        'g6': G6,
        'g7': G7,
        'g8': G8,
        'g9': G9,
        'g10': G10,
        'g11': G11,
        'g12': G12,
        'g13': G13,
        'g14': G14,
        'g15': G15,
        'g16': G16,
        'g17': G17,
        'g18': G18,
        'g19': G19,
        'g20': G20,
        'g21': G21,
        'g22': G22,
        'g23': G23,
        'g24': G24,
        'wfg1': WFG1,
        'wfg2': WFG2,
        'wfg3': WFG3,
        'wfg4': WFG4,
        'wfg5': WFG5,
        'wfg6': WFG6,
        'wfg7': WFG7,
        'wfg8': WFG8,
        'wfg9': WFG9
    }

    if name not in PROBLEM:
        raise Exception("Problem not found.")

    return PROBLEM[name](*args, **kwargs)
