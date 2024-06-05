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

#include <iostream>
#include <vector>

#include "paddle/extension.h"

std::vector<paddle::Tensor> wavefront_lstm_forward(const paddle::Tensor &x, const paddle::Tensor &w, int hidden_size, int num_layers, int time_steps);

std::vector<paddle::Tensor> WavefrontLstmForward(const paddle::Tensor &x, const paddle::Tensor &w, int hidden_size, int num_layers, int time_steps) {
    if (x.is_gpu()) {
        return wavefront_lstm_forward(x, w, hidden_size, num_layers, time_steps);
    } else {
        PD_THROW("Not implemented.");
    }
}

PD_BUILD_OP(wavefront_lstm)
    .Inputs({"X", "W"})
    .Outputs({"Out"})
    .Attrs({"hidden_size: int",
            "num_layers: int",
            "time_steps: int"})
    .SetKernelFn(PD_KERNEL(WavefrontLstmForward));
