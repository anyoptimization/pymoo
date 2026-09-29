"""COMO-CMA-ES — multi-objective CMA-ES via the Sofomore framework (comocma)."""

import numpy as np

from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.docs import parse_doc_string
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.normalization import ZeroToOneNormalization


class COMOCMAES(Algorithm):
    """COMO-CMA-ES (Toure, Auger, Brockhoff & Hansen, GECCO 2019).

    A thin pymoo wrapper around the authors' ``comocma`` package. COMO-CMA-ES
    instantiates the Sofomore framework with a set of CMA-ES kernels that jointly
    maximize the Uncrowded Hypervolume Improvement (UHVI), giving strong anytime
    bi-objective performance — the CMA-ES family that is competitive with the
    documented bbob-biobj winners, which plain MO-CMA-ES is not at small budgets.

    Requires ``comocma`` (``pip install comocma``) and is **bi-objective only**.
    The search runs in a normalized ``[0, 1]`` decision space.

    Anytime behaviour: like CMA-ES-based multi-objective methods generally, it is a
    *late bloomer* — each kernel is a full CMA-ES, so on hard (ill-conditioned or
    multi-modal) landscapes it converges more slowly than population methods at
    small budgets but catches up and ties them given an adequate budget. Prefer a
    generous evaluation budget on difficult problems.
    """

    def __init__(self, pop_size=32, sigma=0.2, reference_point=None, ref_factor=1e6,
                 sampling=FloatRandomSampling(), output=MultiObjectiveOutput(), **kwargs):
        """Initialize COMO-CMA-ES.

        Args:
            pop_size: Number of CMA-ES kernels — roughly the number of Pareto-front
                points sought. Each kernel is a full CMA-ES.
            sigma: Initial step size in the normalized ``[0, 1]`` decision space.
            reference_point: Hypervolume reference point for the UHVI indicator. It is
                fixed for the whole run and should lie just beyond the nadir, as in the
                authors' bbob-biobj benchmark. If ``None`` and the problem knows its
                Pareto front, ``nadir + 0.1 * (nadir - ideal)`` is used. Otherwise a
                large *equal* reference point ``[R, R]`` with ``R = ref_factor * max|F|``
                from the initial sample is used as a fallback; with such a far reference
                point, kernels can stall on the boundary of a bounded objective (for
                example ``f1 = 0`` on ZDT1-3), so pass a reference point near the nadir
                whenever one is known.
            ref_factor: Multiplier for the fallback reference point (see above).
            sampling: Sampling used for the initial population / reference estimate.
            output: Display output used to report progress during the run.
            **kwargs: Additional keyword arguments forwarded to ``Algorithm``.
        """
        super().__init__(output=output, **kwargs)
        self.pop_size = pop_size
        self.sigma0 = sigma
        self.reference_point = reference_point
        self.ref_factor = ref_factor
        self.sampling = sampling
        self.norm = None
        self.moes = None
        self._sols = None

    def _setup(self, problem, **kwargs):
        if problem.n_obj != 2:
            raise ValueError("COMO-CMA-ES (comocma) supports only bi-objective problems.")
        self.norm = ZeroToOneNormalization(problem.xl, problem.xu)

    def _initialize_infill(self):
        return self.sampling(self.problem, self.pop_size, random_state=self.random_state)

    def _initialize_advance(self, infills=None, **kwargs):
        try:
            import comocma
        except ImportError as exc:
            raise ImportError(
                "COMO-CMA-ES requires the comocma package — install it with "
                "`pip install comocma`."
            ) from exc

        n = self.problem.n_var
        rs = self.random_state

        if self.reference_point is not None:
            ref_point = list(self.reference_point)
        else:
            # COMO-CMA-ES approximates the optimal p-distribution of the hypervolume, so
            # the reference point should sit just beyond the nadir. A far reference point
            # gives dominated kernels on a bounded objective a zero UHVI distance, and
            # they stall there (#811).
            try:
                pf = self.problem.pareto_front()
            except OSError:  # the front has to be downloaded and the download failed
                pf = None
            if pf is not None and len(pf) > 0:
                ideal, nadir = pf.min(axis=0), pf.max(axis=0)
                ref_point = (nadir + 0.1 * (nadir - ideal)).tolist()
            else:
                # fallback: a large *equal* reference point balances the two
                # objectives' hypervolume contributions regardless of their scales
                r = self.ref_factor * max(1.0, float(np.abs(infills.get("F")).max()))
                ref_point = [r] * self.problem.n_obj

        x0 = [list(rs.random(n)) for _ in range(self.pop_size)]
        seed = int(rs.integers(0, 2**31 - 1))
        cmas = comocma.get_cmas(
            x0, self.sigma0,
            inopts={"bounds": [[0.0] * n, [1.0] * n], "verbose": -9, "seed": seed},
        )
        # restart converged kernels so the whole budget stays productive and
        # ask("all") never runs out of active kernels (which otherwise asserts)
        self.moes = comocma.Sofomore(
            cmas, reference_point=ref_point,
            opts={"restart": comocma.get_kernel_random_restart},
        )
        self.pop = infills

    def _infill(self):
        # ask("all") advances every kernel each generation (parallel mode) — far
        # more effective than the default sequential single-kernel ask()
        try:
            self._sols = self.moes.ask("all")
        except AssertionError:
            self._sols = []
        if len(self._sols) == 0:  # no active kernels left — end the run
            self.termination.force_termination = True
            return self.pop
        return Population.new(X=self.norm.backward(np.array(self._sols)))

    def _advance(self, infills=None, **kwargs):
        if len(self._sols) == 0:  # run ended in _infill (no active kernels)
            return
        self.moes.tell(self._sols, infills.get("F").tolist())
        Fp = np.array(self.moes.pareto_front_cut, dtype=float)
        Xp = np.array(self.moes.pareto_set_cut, dtype=float)
        # the cut set and front can momentarily desync in length when kernels go
        # inactive — only update the population when they align.
        m = min(len(Fp), len(Xp))
        if m == 0:
            # no kernel dominates the reference point yet: report the kernels'
            # incumbents rather than keeping the initial sample as the result
            told = [k for k in self.moes if k.objective_values is not None]
            Xp = np.array([k.incumbent for k in told], dtype=float)
            Fp = np.array([k.objective_values for k in told], dtype=float)
            m = len(told)
        if m > 0:
            self.pop = Population.new(X=self.norm.backward(Xp[:m]), F=Fp[:m])


parse_doc_string(COMOCMAES.__init__)
