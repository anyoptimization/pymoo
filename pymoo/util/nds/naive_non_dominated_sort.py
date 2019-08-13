from pymoo.util.dominator import Dominator


def naive_non_dominated_sort(F, **kwargs):
    M = Dominator.calc_domination_matrix(F)

    fronts = []
    remaining = set(range(M.shape[0]))

    while len(remaining) > 0:

        front = []

        for i in remaining:

            is_dominated = False
            dominating = set()

            for j in front:
                rel = M[i, j]
                if rel == 1:
                    dominating.add(j)
                elif rel == -1:
                    is_dominated = True
                    break

            if is_dominated:
                continue
            else:
                front = [x for x in front if x not in dominating]
                front.append(i)

        [remaining.remove(e) for e in front]
        fronts.append(front)

    return fronts
