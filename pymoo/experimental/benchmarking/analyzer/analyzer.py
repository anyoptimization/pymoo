from pymoo.experimental.benchmarking.util import filter_by


class Analyzer:

    def __init__(self, default_group_by=None) -> None:
        super().__init__()
        self.default_group_by = default_group_by

    def run(self, result, group_by=None, **kwargs):

        if group_by is not None:
            groups = list(filter_by(result, group_by, return_group=True))
        else:
            groups = [(e, e) for e in result]

        ret = []
        for scope, data in groups:
            vals = self.do(data, scope=scope, **kwargs)
            if vals is not None:
                ret.append({**scope, **vals})
            else:
                ret.extend(data)

        return ret

    def do(self, scope, data, **kwargs):
        pass