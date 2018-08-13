from keras.models import model_from_json


def model_to_dict(model):
    """Turns a Keras model into a Python dictionary

    :param model: Keras model instance
    :return: dictionary with model information
    """
    return dict(model=model.to_json(), weights=model.get_weights())


def dict_to_model(dict):
    """Turns a Python dictionary with model architecture and weights
    back into a Keras model

    :param dict: dictionary with `model` and `weights` keys.
    :return: Keras model instantiated from dictionary
    """
    model = model_from_json(dict['model'])
    model.set_weights(dict['weights'])
    return model
