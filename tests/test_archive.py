import numpy as np

from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.util.archive import SingleObjectiveArchive, MultiObjectiveArchive, SurvivalTruncation
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def test_unconstr_add_to_archive():
    archive = SingleObjectiveArchive()
    assert len(archive) == 0

    a = Individual(X=np.array([5.0]), F=np.array([5.0]))
    pop = Population.create(a)

    archive = archive.add(pop)
    assert len(archive) == 1

    b = Individual(X=np.array([0.0]), F=np.array([0.0]))
    pop = Population.create(b)
    archive = archive.add(pop)

    assert len(archive) == 1
    assert archive[0].f == 0.0

    archive = archive.add(pop)
    assert len(archive) == 1

    c = Individual(X=np.array([-1.0]), F=np.array([0.0]))
    pop = Population.create(c)

    archive = archive.add(pop)
    assert len(archive) == 2


def test_constr_add_to_archive():
    archive = SingleObjectiveArchive()
    assert len(archive) == 0

    a = Individual(X=np.array([0.0]), F=np.array([0.0]), CV=np.array([1.0]))
    pop = Population.create(a)

    archive = archive.add(pop)
    assert len(archive) == 1

    b = Individual(X=np.array([5.0]), F=np.array([5.0]), CV=np.array([0.5]))
    pop = Population.create(b)
    archive = archive.add(pop)

    assert len(archive) == 1
    assert archive[0].f == 5.0
    assert archive[0].cv == 0.5

    c = Individual(X=np.array([10.0]), F=np.array([10.0]), CV=np.array([0.0]))
    pop = Population.create(c)
    archive = archive.add(pop)

    assert len(archive) == 1
    assert archive[0].f == 10.0
    assert archive[0].cv == 0.0

    d = Individual(X=np.array([7.0]), F=np.array([7.0]), CV=np.array([0.0]))
    pop = Population.create(d)
    archive = archive.add(pop)

    assert len(archive) == 1
    assert archive[0].f == 7.0
    assert archive[0].cv == 0.0


def test_max_size():
    archive = SingleObjectiveArchive(max_size=1)
    assert len(archive) == 0

    a = Individual(X=np.array([5.0]), F=np.array([0.0]))
    pop = Population.create(a)

    archive = archive.add(pop)
    assert len(archive) == 1

    c = Individual(X=np.array([-1.0]), F=np.array([0.0]))
    pop = Population.create(c)

    archive = archive.add(pop)
    assert len(archive) == 1


def test_multi_objective_archive():
    a = Individual(X=np.array([5.0]), F=np.array([1.0, 5.0]))
    pop = Population.create(a)

    archive = MultiObjectiveArchive().add(pop)
    assert len(archive) == 1

    b = Individual(X=np.array([10.0]), F=np.array([2.0, 2.0]))
    pop = Population.create(b)

    archive = archive.add(pop)
    assert len(archive) == 2


def test_multi_objective_archive_multi():
    np.random.seed(1)
    X, F = np.random.random(size=(100, 10)), np.random.random(size=(100, 3))
    pop = Population.new(X=X, F=F)

    archive = MultiObjectiveArchive().add(pop)
    actual = pop[NonDominatedSorting().do(F, only_non_dominated_front=True)]
    assert np.all(actual == archive)

    archive = MultiObjectiveArchive(max_size=5).add(pop)
    assert len(archive) == 5

    archive = MultiObjectiveArchive(max_size=5, truncation=SurvivalTruncation(RankAndCrowdingSurvival())).add(pop)
    assert len(archive) == 5

