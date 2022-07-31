//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void myproject(
    hls::stream<input_t> &em_barrel,
    hls::stream<result_t> &layer55_out,
    model_default_t s3[4],
    model_default_t b3[4],
    weight4_t w4[1600],
    bias4_t b4[16],
    model_default_t s6[16],
    model_default_t b6[16],
    weight9_t w9[4608],
    bias9_t b9[32],
    model_default_t s11[32],
    model_default_t b11[32],
    weight13_t w13[9216],
    bias13_t b13[32],
    model_default_t s15[32],
    model_default_t b15[32],
    weight18_t w18[18432],
    bias18_t b18[64],
    model_default_t s20[64],
    model_default_t b20[64],
    weight22_t w22[36864],
    bias22_t b22[64],
    model_default_t s24[64],
    model_default_t b24[64],
    weight27_t w27[73728],
    bias27_t b27[128],
    model_default_t s29[128],
    model_default_t b29[128],
    weight31_t w31[147456],
    bias31_t b31[128],
    model_default_t s33[128],
    model_default_t b33[128],
    weight36_t w36[294912],
    bias36_t b36[256],
    model_default_t s38[256],
    model_default_t b38[256],
    weight40_t w40[589824],
    bias40_t b40[256],
    model_default_t s42[256],
    model_default_t b42[256],
    weight45_t w45[589824],
    bias45_t b45[256],
    model_default_t s47[256],
    model_default_t b47[256],
    weight49_t w49[65536],
    bias49_t b49[256],
    model_default_t s51[256],
    model_default_t b51[256],
    weight53_t w53[256],
    bias53_t b53[1],
    unsigned short &const_size_in_1,
    unsigned short &const_size_out_1
);

#endif
