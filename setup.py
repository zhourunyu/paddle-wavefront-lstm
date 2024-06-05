from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name='paddle_custom_ops',
    ext_modules=CUDAExtension(
        sources=['wavefront_lstm_op.cc', 'wavefront_lstm_op.cu'],
        extra_compile_args={'cc': ['-w'], 'nvcc': ['-O3']},
        libraries=['cudart', 'cuda'],
    )
)