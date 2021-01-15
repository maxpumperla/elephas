from tensorflow.keras.models import model_from_json, Model


def model_to_dict(model: Model):
    """Turns a Keras model into a Python dictionary

    :param model: Keras model instance
    :return: dictionary with model information
    """
    return dict(model=model.to_json(), weights=model.get_weights())


def dict_to_model(_dict: dict):
    """Turns a Python dictionary with model architecture and weights
    back into a Keras model

    :param _dict: dictionary with `model` and `weights` keys.
    :return: Keras model instantiated from dictionary
    """
    model = model_from_json(_dict['model'])
    model.set_weights(_dict['weights'])
    return model
