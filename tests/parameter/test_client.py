from unittest.mock import patch

import pytest

from elephas.parameter import BaseParameterClient, HttpClient, SocketClient


@pytest.mark.parametrize('client_type, obj', [('http', HttpClient),
                                             ('socket', SocketClient)])
def test_client_factory_method(client_type, obj):
    with patch('elephas.parameter.client.socket'):
        assert type(BaseParameterClient.get_client(client_type, 4000)) == obj
