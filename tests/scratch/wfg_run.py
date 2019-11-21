from pymoo.factory import get_problem



for m in range(2, 15, 3):
    
    for n in range(6, 24, 6):
    
        if m == 2:
            k = 4
            l = 2
        else:
            k = 2 * (m - 1)
            l = 
    
        problems = [get_problem("wfg" % k, ) for k in range(1, 10)]
