from random import shuffle
import numpy as np

from rand.my_random_generator import MyRandomGenerator
from util.quicksort import quicksort

if __name__ == '__main__':
    r = MyRandomGenerator()
    r.seed(0.1)
    print(r.random())
    print(r.random())
    print(r.random())


    for i in range(100):
        print(i)
        l = list(range(100))
        #shuffle(l)
        second = np.random.random(100)

        quicksort(l, key=lambda x:second[x], rnd=MyRandomGenerator())
        print(l)
        print([second[i] for i in l])
