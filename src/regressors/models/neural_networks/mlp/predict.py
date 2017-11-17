

def predict(model, data, weights=None, model_weights=None):

    x_test = data['testing_data'][0]

    if weights is not None:
        pass
    elif model_weights is not None:
        pass
    else:
        predicted = model.predict(x_test)

    return predicted