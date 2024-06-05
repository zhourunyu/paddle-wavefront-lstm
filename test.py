import paddle
import time
from paddle_custom_ops import wavefront_lstm

def read_tensor_from_file(file_name: str) -> paddle.Tensor:
    with open(file_name, 'r') as f:
        buf = f.read()
        
    buf = buf.split()
    data = list(map(float, buf))
    return paddle.to_tensor(data, dtype=paddle.float32, place="gpu")

input_size = 256
hidden_size = 256
time_steps = 100
num_layers = 10

class WavefrontLSTM(paddle.nn.Layer):
    def __init__(self, input_size, hidden_size, num_layers, time_steps):
        super(WavefrontLSTM, self).__init__()
        self.state = paddle.empty([num_layers, time_steps + 6, hidden_size], dtype=paddle.float32)
        self.model = None

    def load_model(self, model):
        self.model = model

    def forward(self, input):
        return wavefront_lstm(input, self.model, self.state)

def benchmark(model, input, name):
    warmup = 10
    iterations = 100
    for _ in range(warmup):
        model(input)
    paddle.device.synchronize()
    start = time.time()
    for _ in range(iterations):
        model(input)
    paddle.device.synchronize()
    end = time.time()
    print("{} time: {} ms".format(name, (end - start) / iterations * 1000))

if __name__ == '__main__':
    paddle.set_device('gpu')
    input = read_tensor_from_file('test_data/input_params.txt').reshape([time_steps, input_size])
    model = read_tensor_from_file('test_data/model_params.txt')
    expect_results = read_tensor_from_file('test_data/expect_results.txt').reshape([time_steps, hidden_size])

    lstm_wavefront = WavefrontLSTM(input_size, hidden_size, num_layers, time_steps)
    lstm_wavefront.load_model(model)
    output = lstm_wavefront(input)
    if paddle.allclose(output, expect_results, atol=1e-5, rtol=1e-5):
        print("Test passed!")
    else:
        print("Test failed!")

    lstm_paddle = paddle.nn.LSTM(input_size, hidden_size, num_layers)
    benchmark(lstm_wavefront, input, "Wavefront LSTM")
    benchmark(lstm_paddle, input.unsqueeze(0), "Paddle LSTM")
