from keras.models import model_from_json


def model_to_dict(model):
    return dict(model=model.to_json(), weights=model.get_weights())


def dict_to_model(dict):
    model = model_from_json(dict['model'])
    model.set_weights(dict['weights'])
    return model
