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
    hls::stream<result_t> &layer46_out,
    /*
    model_default_t s3[4],
    model_default_t b3[4],
    weight4_t w4[1600],
    bias4_t b4[16],
    weight8_t w8[4608],
    bias8_t b8[32],
    weight11_t w11[9216],
    bias11_t b11[32],
    weight15_t w15[18432],
    bias15_t b15[64],
    weight18_t w18[36864],
    bias18_t b18[64],
    weight22_t w22[73728],
    bias22_t b22[128],
    weight25_t w25[147456],
    bias25_t b25[128],
    weight29_t w29[294912],
    bias29_t b29[256],
    weight32_t w32[589824],
    bias32_t b32[256],
    weight36_t w36[589824],
    bias36_t b36[256],
    model_default_t s38[256],
    model_default_t b38[256],
    weight40_t w40[65536],
    bias40_t b40[256],
    model_default_t s42[256],
    model_default_t b42[256],
    weight44_t w44[256],
    bias44_t b44[1],
    */
    unsigned short &const_size_in_1,
    unsigned short &const_size_out_1
);

#endif
