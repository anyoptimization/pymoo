import numpy
import tensorflow as tf

from metamodels.metamodel import MetaModel


class TFLearnMetamodel(MetaModel):


    def __init__(self):
        super().__init__()
        self.estimators = None

    def _get_parameter(self, d={}):
        n_var = d['n_var']
        return [None]

    def _predict(self, metamodel, X):

        vals = []
        for estimator in self.estimators:
            predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": X},
                num_epochs=1,
                shuffle=False)

            val = estimator.predict(input_fn=predict_input_fn)
            val = [e for e in list(val)]
            vals.append(numpy.array([e['predictions'] for e in val]).T[0])

        vals = numpy.array(vals)

        return numpy.mean(vals, axis=0), numpy.std(vals, axis=0)

    def _create_and_fit(self, parameter, X, F, expensive=False):

        self.estimators = []
        feature_columns = [tf.feature_column.numeric_column("x", shape=[X.shape[1]])]
        input_fn = tf.estimator.inputs.numpy_input_fn({"x": X}, F, batch_size=4, num_epochs=None, shuffle=True)

        for _ in range(10):
            estimator = tf.estimator.DNNRegressor([1024, 512, 256], feature_columns=feature_columns)
            estimator = estimator.train(input_fn=input_fn, steps=1000)
            self.estimators.append(estimator)


