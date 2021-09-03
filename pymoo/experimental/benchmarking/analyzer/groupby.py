from pymoo.experimental.benchmarking.analyzer.analyzer import Analyzer

DEFAULT_FUNCTIONS = {
    "first": lambda x: x[0],
    "collect": lambda x: x,
}


class GroupBy(Analyzer):

    def __init__(self,
                 attrs,
                 funcs={},
                 **kwargs) -> None:
        """
        This is a result analyzer which groups elements and aggregates attributes.
        Similar to database queries but on a Python dictionary level.

        Parameters
        ----------
        attrs : tuple
            A tuple defining (in,func,out) which is an aggregation of one attribute.

        funcs : dict
            If some other functions should be injected to be known for the analysis.

        """
        super().__init__(**kwargs)
        self.attrs = attrs
        self.funcs = {**DEFAULT_FUNCTIONS, **funcs}

    def do(self, data, **kwargs):
        ret = {}

        for elems in self.attrs:

            # parse the attribute inputs
            if len(elems) == 2:
                _in, _func = elems
                _out = f"{str(_func)}({_in})"
            elif len(elems) == 3:
                _in, _func, _out = elems
            else:
                raise Exception("The group by attributes are either (in, func) or (in,func,out).")

            if isinstance(_func, str):
                if _func not in self.funcs:
                    raise Exception(f"Unknown Function: {_func}")
                _func = self.funcs[_func]

            # get the data from the current data set
            vals = [entry[_in] for entry in data]

            # apply the function to the input
            aggr = _func(vals)

            # finally store the output dta
            ret[f"{_out}"] = aggr

        return ret
