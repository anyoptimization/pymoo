from pymoo.core.population import Population


def test_has_method():
    pop = Population(100)
    assert pop.has("X")
    assert not pop.has("opt")

    pop[:-1].set("opt", False)
    assert not pop.has("opt")

    pop[-1].set("opt", True)
    assert pop.has("opt")


