__author__ = 'tmorales'

def splitTimeSerie(features, tsLabels, periods):
    """

    :param ts: timeserie
    :param periods: periods split timeserie (dict)
    :return:
    """
    if len(periods) == 3:
        # --- labels
        y_train = tsLabels[periods['train'][0]:periods['train'][1]]
        y_test = tsLabels[periods['test'][0] : periods['test'][1]]
        y_forecast = tsLabels[periods['forecast'][0] : periods['forecast'][1]]

        # --- features
        x_train = features[:len(y_train)]
        x_test = features[len(y_train):len(y_train) + len(y_test)]
        x_forecast = features[len(y_train) + len(y_test):len(y_train) + len(y_test) + len(y_forecast)]

    if len(periods) == 2:
        # no implementado
        pass

    return x_train, y_train, x_test, y_test, x_forecast, y_forecast