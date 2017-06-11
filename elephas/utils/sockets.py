from six.moves import cPickle as pickle
from socket import gethostbyname, gethostname


def determine_master(port=':5000'):
    return gethostbyname(gethostname()) + port


def receive_all(socket, num_bytes):
    """Reads `num_bytes` bytes from the specified socket.
    # Arguments
        socket: Open socket.
        num_bytes: Number of bytes to read.
    """
    buffer = ''
    buffer_size = 0
    bytes_left = num_bytes
    while buffer_size < num_bytes:
        data = socket.recv(bytes_left)
        delta = len(data)
        buffer_size += delta
        bytes_left -= delta
        buffer += data
    return buffer


def receive(socket, num_bytes=20):
    """Fetch data frame from open socket
    # Arguments
        socket: Open socket.
        num_bytes: Number of bytes to read.
    """
    length = int(receive_all(socket, num_bytes).decode())
    serialized_data = receive_all(socket, length)
    return pickle.loads(serialized_data)


def send(socket, data, num_bytes=20):
    """Send data to specified socket.
    # Arguments
        socket: socket. Opened socket.
        data: any. Data to send.
    """
    pickled_data = pickle.dumps(data, -1)
    length = str(len(pickled_data)).zfill(num_bytes)
    socket.sendall(length.encode())
    socket.sendall(pickled_data)
