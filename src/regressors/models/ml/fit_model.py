

def fit_models(model, data):

    print('i am at --ml: fit_model')
    # -- load data
    x_train, y_train = data['training_data']

    # -- run model
    model = model.fit(x_train, y_train)

    return model