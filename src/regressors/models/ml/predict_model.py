
def predict_models(model, data):

    x_test = data['testing_data'][0]

    predicted = model. predict(x_test)

    return predicted
