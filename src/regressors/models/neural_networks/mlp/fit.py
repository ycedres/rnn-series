

def fit(model, data):

    print ('i am at --mlp: fit_model')

    epochs = 10
    batch_size = 20

    # -- load training data
    x_train, y_train = data['training_data']
    # -- load validation data
    x_val, y_val = data['validation_data']


    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)



    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        )
    #plot history

    return model
