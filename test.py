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
        self.weight = None

    def load_model(self, weight):
        self.weight = weight

    def forward(self, input):
        return wavefront_lstm(input, self.weight, self.state)

def convert_weight(weight_dict):
    size_per_layer = 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
    weight = paddle.empty([num_layers * size_per_layer], dtype=paddle.float32)
    for i in range(num_layers):
        weight_ih = weight_dict['weight_ih_l' + str(i)].reshape([4, hidden_size, input_size])
        weight_hh = weight_dict['weight_hh_l' + str(i)].reshape([4, hidden_size, hidden_size])
        bias_ih = weight_dict['bias_ih_l' + str(i)].reshape([4, hidden_size])
        bias_hh = weight_dict['bias_hh_l' + str(i)].reshape([4, hidden_size])

        weight_ih = paddle.transpose(weight_ih, [0, 2, 1])
        weight_hh = paddle.transpose(weight_hh, [0, 2, 1])
        bias = bias_ih + bias_hh
        weight[i * size_per_layer:(i + 1) * size_per_layer] = paddle.concat([weight_ih.flatten(), weight_hh.flatten(), bias.flatten()])

    return weight

def benchmark(model, input, name):
    warmup = 10
    iterations = 1000
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

    lstm_paddle = paddle.nn.LSTM(input_size, hidden_size, num_layers)
    lstm_paddle.eval()
    lstm_wavefront = WavefrontLSTM(input_size, hidden_size, num_layers, time_steps)
    state_dict = lstm_paddle.state_dict(include_sublayers=False)
    input_data = paddle.randn([time_steps, input_size], dtype=paddle.float32)
    lstm_wavefront.load_model(convert_weight(state_dict))

    output_wavefront = lstm_wavefront(input_data)
    output_paddle, _ = lstm_paddle(input_data.unsqueeze(0))
    output_paddle = output_paddle.squeeze(0)
    if paddle.allclose(output_wavefront, output_paddle, atol=1e-3, rtol=1e-3):
        print("Test passed!")
    else:
        print("Test failed!")

    benchmark(lstm_wavefront, input_data, "Wavefront LSTM")
    benchmark(lstm_paddle, input_data.unsqueeze(0), "Paddle LSTM")
