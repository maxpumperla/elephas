import pytest

from elephas.utils.model_utils import ModelType, LossModelTypeMapper


@pytest.mark.parametrize('loss, model_type', [('binary_crossentropy', ModelType.CLASSIFICATION),
                                              ('mean_squared_error', ModelType.REGRESSION),
                                              ('categorical_crossentropy', ModelType.CLASSIFICATION),
                                              ('mean_absolute_error', ModelType.REGRESSION)])
def test_model_type_mapper(loss, model_type):
    assert LossModelTypeMapper().get_model_type(loss) == model_type


def test_model_type_mapper_custom():
    LossModelTypeMapper().register_loss('test', ModelType.REGRESSION)
    assert LossModelTypeMapper().get_model_type('test') == ModelType.REGRESSION


def test_model_type_mapper_custom_callable():
    def custom_loss(y_true, y_pred):
        return y_true - y_pred
    LossModelTypeMapper().register_loss(custom_loss, ModelType.REGRESSION)
    assert LossModelTypeMapper().get_model_type('custom_loss') == ModelType.REGRESSION