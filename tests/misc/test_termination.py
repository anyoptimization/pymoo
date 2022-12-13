from pymoo.termination.max_time import TimeBasedTermination


def test_time_based_termination_more_than_one_day():
    termination = TimeBasedTermination("99:00:00")
    assert termination.max_time == 356400

    termination = TimeBasedTermination("105:10:00")
    assert termination.max_time == 378600

    termination = TimeBasedTermination("40")
    assert termination.max_time == 40

    termination = TimeBasedTermination("1:40")
    assert termination.max_time == 100
