from pymoo.algorithms.nsga3 import NSGA3
from pymoo.emo.maximum_of_extremes import MaximumOfExtremesReferenceLineSurvival
from pymoo.emo.maximum_of_non_dominated import MaximumOfNonDominatedReferenceLineSurvival
from pymoo.emo.maximum_of_population import MaximumOfPopulationReferenceSurvival
from pymoo.emo.propose import ProposeReferenceLineSurvival
from pymoo.experimental.normalization.perfect import PerfectReferenceLineSurvival


def get_algorithm(name, ref_dirs, *args):

    a = NSGA3(ref_dirs)

    if name == "max-of-pop":
        a.survival = MaximumOfPopulationReferenceSurvival(ref_dirs)
    elif name == "max-non-dom":
        a.survival = MaximumOfNonDominatedReferenceLineSurvival(ref_dirs)
    elif name == "max-of-extremes":
        a.survival = MaximumOfExtremesReferenceLineSurvival(ref_dirs)
    elif name == "propose":
        a.survival = ProposeReferenceLineSurvival(ref_dirs)
    elif name == "test":
        a.survival = MaximumOfExtremesReferenceLineSurvival(ref_dirs)
    elif name == "perfect":
        a.survival = PerfectReferenceLineSurvival(ref_dirs, *args)
    elif name == "original":
        pass
    else:
        raise Exception("Unknown algorithm proposed for EMO2019: %s" % name)

    return a