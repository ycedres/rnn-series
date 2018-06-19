

# Invoker
class Runner(object):
    def __init__(self):
        self._queue = []

    #def history():
    #    return self._history

    def local_run(operation): #la operacion tiene al modelo como atributo
        #self._history = self._history + (operation,)
        #self._queue.append(operation)
        operation.execute()

class LocalRunner(Runner):
    def __init__(self):
        self._queue = []

    def append(operation):
        self._queue.append(operation)

    def run_queue(operation):
        pass

    def run_operation(operation):
        operation.excecute()

# Command


class Operation(object):
    def __init__(self, model):
        self._model = model

# Specific Commands


class TrainOperation(Operation):

    def __init__(self, model):
        self._model = model

    def run(self, train_features,
                  train_target,
                  validation_features,
                  validation_target):

        self._model.train(train_features, train_target, validation_features,
                          validation_target)


class TestOperation(Operation):

    def __init__(self, model):
        self._model = model

    def run(self, test_features):
        return self._model.test(test_features)
