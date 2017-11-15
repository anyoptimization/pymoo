import numpy as np
import pygmo as pg
import plotly
import plotly.graph_objs as go


from measures.hypervolume import Hypervolume
from measures.igd import IGD
from model.individual import Individual
from util.misc import normalize
from util.rank_and_crowding import RankAndCrowdingSurvival

if __name__ == '__main__':

    exercise = 3

    if exercise is 4:

        task = 'b'

        N = [50, 100, 500]
        M = [2, 3, 4, 5, 8, 10, 15]

        data = np.zeros((len(N), len(M)))

        for i, n in enumerate(N):

            for j, m in enumerate(M):

                val = []
                for _ in range(50):
                    f = np.random.random((n, m))
                    fronts = pg.fast_non_dominated_sorting(f)[0]

                    if task == 'a':
                        val.append(len(fronts[0]))
                    elif task == 'b':
                        val.append(len(fronts))
                    else:
                        raise ValueError

                data[i, j] = np.median(val)

        trace0 = go.Scatter(
            y=data[0,:],
            x=M,
            mode='lines+markers',
            name='50'
        )
        trace1 = go.Scatter(
            y=data[1, :],
            x=M,
            mode='lines+markers',
            name='100'
        )
        trace2 = go.Scatter(
            y=data[2, :],
            x=M,
            mode='lines+markers',
            name='500'
        )

        title= "number of non-dominated points" if task == 'a' else "number of fronts"

        layout = dict(title='',
                      xaxis=dict(title='M'),
                      yaxis=dict(title=title),
                      )

        plotly.offline.plot(
            {
                "data": [trace0, trace1, trace2],
                "layout" : layout
            },
            filename="ha.html"
        )

        print(data)

        np.savetxt("HA6_%s.csv" % task, data, delimiter=",")

    elif exercise is 1:

        names = ['2', '3', 'b', 'd', 'e', '5', 'a', '4', '1', 'c']

        data = np.array(
            [[1, -1], [1.5, 0], [2.2, -0.5], [3.0, -3.0], [3.0, -1.5], [3.5, -2.0], [3.5, 0], [4.5, -1.0], [5.0, -2.5],
             [5.0, -2.0]])
        pop = [Individual(f=x) for x in data]

        rank, crowding = RankAndCrowdingSurvival.calc_rank_and_crowding(pop)

        for i in sorted(range(len(pop)), key=lambda x: (rank[x], -crowding[x])):
            print(names[i], pop[i].f, rank[i], crowding[i])

    elif exercise is 2:

        F = np.array([[1, 10], [2, 7], [3, 6], [5, 2], [8, 1], ])
        F_norm = normalize(F, np.min(F, axis=0), np.array([11, 11]))

        print(Hypervolume(reference_point=(11, 11)).calc(F))
        print(Hypervolume(reference_point=(1.0, 1.0)).calc(F_norm))

    elif exercise is 3:

        front = np.array([[1, 12], [2, 9], [3, 8], [4, 7], [5, 5], [6, 4], [8, 3], [10, 2], [12, 1]])
        F = np.array([[3,10], [4,9], [6,7], [7,5], [10,3]])

        front_min = np.min(front, axis=0)
        front_max = np.max(front, axis=0)

        front_norm = normalize(front, front_min, front_max)
        F_norm = normalize(F, front_min, front_max)

        print(IGD(front).calc(F))

        print(IGD(front_norm).calc(F_norm))
