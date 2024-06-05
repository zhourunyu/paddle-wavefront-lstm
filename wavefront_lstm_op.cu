// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/extension.h"
#include "cuda.h"
#include "cuda_runtime.h"

#define CHECK_GPU_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")

enum LSTMScaleParams {
    kLstmGateNumber = 4,
    kHiddenSize = 256,
    kInputSize = 256,
    kCellNumber = 10,
    kLstmTimestep = 100,
};

enum LSTMKernelScaleParams {
    kThreadsPerWarp = 32,
    kWarpsPerBlock = 8,
    kColumnsPerBlock = kThreadsPerWarp,
    kGemvBlockNumber = kHiddenSize / kColumnsPerBlock,
    kRowsPerWarp = kHiddenSize / kWarpsPerBlock,
};

struct CellModel {
    float weights_w[kLstmGateNumber][kInputSize][kHiddenSize];
    float weights_u[kLstmGateNumber][kHiddenSize][kHiddenSize];
    float bias[kLstmGateNumber][kHiddenSize];
};
static_assert(sizeof(CellModel) == sizeof(CellModel::weights_w) +
                                   sizeof(CellModel::weights_u) +
                                   sizeof(CellModel::bias),
              "Expect the data to be placed continuously.");

#pragma pack(push, 1)
struct ModelParams {
    CellModel cell_model[kCellNumber];
};
#pragma pack(pop)

struct InputData {
    float data[kLstmTimestep][kInputSize];
};

struct CellState {
    float data[kHiddenSize];
};

struct CellTemp {
    float data[kLstmGateNumber][kHiddenSize];
};

#pragma pack(push, 1)
struct CellParams {
    CellState cell_state_h[kCellNumber][kLstmTimestep + 1];
    CellState cell_state_c[kCellNumber];
    CellTemp cell_temp[kCellNumber];
};
#pragma pack(pop)

class WavefrontLSTM {
public:
    explicit WavefrontLSTM(const float *src_model, cudaStream_t stream = 0);
    bool Initialize(const float *input, float *state);
    void Solve();
    bool Fetch(float *output);

private:
    cudaStream_t stream_;
    ModelParams *d_model_params_;
    CellParams *d_cell_params_;
    InputData *d_input_data_;
    float *d_output_;
};

__global__ void wave_compute(ModelParams *d_model_params, CellParams *d_cell_params, 
                             InputData *d_input_data, int step_start_num, int layer_start_num);

WavefrontLSTM::WavefrontLSTM(const float *src_model, cudaStream_t stream) {
    d_model_params_ = (ModelParams *)src_model;
    stream_ = stream;
}

bool WavefrontLSTM::Initialize(const float *input, float *state) {
    d_cell_params_ = (CellParams *)state;
    cudaMemsetAsync((void *)d_cell_params_, 0, sizeof(CellParams::cell_state_h) + sizeof(CellParams::cell_state_c), stream_);
    d_input_data_ = (InputData *)input;
    d_output_ = (float *)d_cell_params_ + kCellNumber * (kLstmTimestep + 1) * kHiddenSize -
                kLstmTimestep * kHiddenSize;
    return true;
}

void WavefrontLSTM::Solve() {
    const int max_wave_size = std::min(kCellNumber, kLstmTimestep);
    const int max_wave_number = kCellNumber + kLstmTimestep - 1;

    for (int wave_idx = 1; wave_idx <= max_wave_number; ++wave_idx) {
        int wave_size = (wave_idx < std::max(kCellNumber, kLstmTimestep))
                        ? std::min(wave_idx, max_wave_size)
                        : (max_wave_size -
                           (wave_idx - std::max(kCellNumber, kLstmTimestep)));
        int step_start_num = (wave_idx < kLstmTimestep) ? wave_idx : kLstmTimestep;
        int layer_start_num =
            (wave_idx < kLstmTimestep) ? 0 : (wave_idx - kLstmTimestep);

        wave_compute<<<kGemvBlockNumber * wave_size, kHiddenSize, 0, stream_>>>(
            d_model_params_, d_cell_params_, d_input_data_, 
            step_start_num, layer_start_num);
    }
}

bool WavefrontLSTM::Fetch(float *output) {
    cudaStreamSynchronize(stream_);
    cudaMemcpy((void *)output, d_output_, sizeof(CellState) * kLstmTimestep,
               cudaMemcpyDeviceToDevice);
    return true;
}

__device__ static inline float sigmoid(float x) {
    return 1.000000e+00f / (1.000000e+00f + __expf(0.000000e+00f - x));
}

__global__ void __launch_bounds__(256, 4)
wave_compute(ModelParams *d_model_params, CellParams *d_cell_params, 
             InputData *d_input_data, int step_start_num, int layer_start_num) {
    const int cell_idx = layer_start_num + blockIdx.x / kGemvBlockNumber;
    const int step_idx = step_start_num - blockIdx.x / kGemvBlockNumber;
    float *d_input = (cell_idx == 0)? &d_input_data->data[step_idx - 1][0]: d_cell_params->cell_state_h[cell_idx - 1][step_idx].data;
    CellState *d_input_state_h =
        &d_cell_params->cell_state_h[cell_idx][step_idx - 1];
    CellState *d_output_state_h =
        &d_cell_params->cell_state_h[cell_idx][step_idx];
    CellState *d_state_c = &d_cell_params->cell_state_c[cell_idx];
    CellTemp *d_temp = &d_cell_params->cell_temp[cell_idx];
    CellModel *d_model = &d_model_params->cell_model[cell_idx];

    const int warp_idx = threadIdx.x / kThreadsPerWarp;
    const int lane_idx = threadIdx.x % kThreadsPerWarp;
    const int col_idx =
        (blockIdx.x % kGemvBlockNumber) * kColumnsPerBlock + lane_idx;

    if (warp_idx == 0) {
        for (int i = 0; i < kLstmGateNumber; ++i) {
            d_temp->data[i][col_idx] = 0.000000e+00f;
        }
    }
    __syncthreads();

    float temp[kLstmGateNumber] = {0.000000e+00f, 0.000000e+00f, 0.000000e+00f,
                                   0.000000e+00f};
    const int row_start_idx = kRowsPerWarp * warp_idx;
    const int row_end_idx = row_start_idx + kRowsPerWarp;
    for (int row_idx = row_start_idx; row_idx < row_end_idx; ++row_idx) {
        float input_data = d_input[row_idx];
        float state_h_data = d_input_state_h->data[row_idx];
        for (int i = 0; i < kLstmGateNumber; ++i) {
            temp[i] =
                fma(d_model->weights_w[i][row_idx][col_idx], input_data, temp[i]);
        }
        for (int i = 0; i < kLstmGateNumber; ++i) {
            temp[i] =
                fma(d_model->weights_u[i][row_idx][col_idx], state_h_data, temp[i]);
        }
    }

    for (int i = 0; i < kLstmGateNumber; ++i) {
        atomicAdd(&d_temp->data[i][col_idx], temp[i]);
    }
    __syncthreads();

    if (warp_idx == 0) {
        float input_gate_x = d_temp->data[0][col_idx] + d_model->bias[0][col_idx];
        float forget_gate = d_temp->data[1][col_idx] + d_model->bias[1][col_idx];
        float input_gate_y = d_temp->data[2][col_idx] + d_model->bias[2][col_idx];
        float output_gate = d_temp->data[3][col_idx] + d_model->bias[3][col_idx];
        input_gate_x = sigmoid(input_gate_x);
        input_gate_y = tanh(input_gate_y);
        output_gate = sigmoid(output_gate);
        forget_gate =
            sigmoid(forget_gate) * d_state_c->data[col_idx];
        d_state_c->data[col_idx] = fma(input_gate_x, input_gate_y, forget_gate);
        d_output_state_h->data[col_idx] =
            (tanh(d_state_c->data[col_idx])) * output_gate;
    }
}

std::vector<paddle::Tensor> wavefront_lstm_forward(const paddle::Tensor &x, const paddle::Tensor &w, paddle::Tensor &state) {
    CHECK_GPU_INPUT(x);
    PD_CHECK(x.place() == paddle::DefaultGPUPlace());

    auto out = paddle::empty({kLstmTimestep, kHiddenSize}, x.type(), x.place());
    const auto &dtype = x.type();
    cudaStream_t stream = x.stream();
    switch (dtype) {
    case ::paddle::DataType::FLOAT32: {
        auto x_data = x.data<float>();
        auto w_data = w.data<float>();
        auto out_data = out.data<float>();
        auto state_data = state.data<float>();

        auto network = new WavefrontLSTM(w_data, stream);
        network->Initialize(x_data, state_data);
        network->Solve();
        network->Fetch(out_data);
    }
    break;
    default:
        PD_THROW("wavefront_lstm_forward not implemented for data type `", dtype, "`");
    }
    return {out};
}
