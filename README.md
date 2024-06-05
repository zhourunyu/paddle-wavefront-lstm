# Wavefront LSTM for Paddle
Wavefront LSTM algorithm, implemented as Paddle C++ Extension.

## Usage
Build and install wheel for custom op
```shell
python3 setup.py install
```
Run test script 
```shell
$ python3 test.py
Test passed!
Wavefront LSTM time: 0.8911538124084473 ms
Paddle LSTM time: 7.381469964981079 ms
```

## Note
The wavefront-lstm op only support fixed shapes and timesteps for now.
