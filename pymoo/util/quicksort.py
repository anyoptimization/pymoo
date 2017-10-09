from rand.default_random_generator import DefaultRandomGenerator


def quicksort(l, key, rnd=DefaultRandomGenerator()):
    quicksort_rec(l, key, 0, len(l) - 1, rnd)
    return list(l)


def swap(l, i, j):
    temp = l[i]
    l[i] = l[j]
    l[j] = temp


def quicksort_rec(l, key, left, right, rnd):
    if left < right:

        index = rnd.randint(left, right)
        swap(l, right, index)

        pivot = key(l[right])
        i = left - 1

        for j in range(left, right):
            if key(l[j]) <= pivot:
                i += 1
                swap(l, i, j)

        index = i + 1
        swap(l, index, right)

        quicksort_rec(l, key, left, index - 1, rnd)
        quicksort_rec(l, key, index + 1, right, rnd)
