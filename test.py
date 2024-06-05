import paddle
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

if __name__ == '__main__':
    input = read_tensor_from_file('test_data/input_params.txt').reshape([time_steps, input_size])
    model = read_tensor_from_file('test_data/model_params.txt')
    expect_results = read_tensor_from_file('test_data/expect_results.txt').reshape([time_steps, hidden_size])
    output = wavefront_lstm(input, model, input_size, hidden_size, time_steps)
    print(paddle.allclose(output, expect_results, atol=1e-5, rtol=1e-5))
