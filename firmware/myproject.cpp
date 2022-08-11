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
#include <iostream>

#include "myproject.h"
#include "parameters.h"

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
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=em_barrel,layer46_out 
    #pragma HLS DATAFLOW 

    const_size_in_1 = N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1;
    const_size_out_1 = N_LAYER_44;

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<model_default_t, 4>(s3, "s3.txt");
        nnet::load_weights_from_txt<model_default_t, 4>(b3, "b3.txt");
        nnet::load_weights_from_txt<weight4_t, 1600>(w4, "w4.txt");
        nnet::load_weights_from_txt<bias4_t, 16>(b4, "b4.txt");
        nnet::load_weights_from_txt<weight8_t, 4608>(w8, "w8.txt");
        nnet::load_weights_from_txt<bias8_t, 32>(b8, "b8.txt");
        nnet::load_weights_from_txt<weight11_t, 9216>(w11, "w11.txt");
        nnet::load_weights_from_txt<bias11_t, 32>(b11, "b11.txt");
        nnet::load_weights_from_txt<weight15_t, 18432>(w15, "w15.txt");
        nnet::load_weights_from_txt<bias15_t, 64>(b15, "b15.txt");
        nnet::load_weights_from_txt<weight18_t, 36864>(w18, "w18.txt");
        nnet::load_weights_from_txt<bias18_t, 64>(b18, "b18.txt");
        nnet::load_weights_from_txt<weight22_t, 73728>(w22, "w22.txt");
        nnet::load_weights_from_txt<bias22_t, 128>(b22, "b22.txt");
        nnet::load_weights_from_txt<weight25_t, 147456>(w25, "w25.txt");
        nnet::load_weights_from_txt<bias25_t, 128>(b25, "b25.txt");
        nnet::load_weights_from_txt<weight29_t, 294912>(w29, "w29.txt");
        nnet::load_weights_from_txt<bias29_t, 256>(b29, "b29.txt");
        nnet::load_weights_from_txt<weight32_t, 589824>(w32, "w32.txt");
        nnet::load_weights_from_txt<bias32_t, 256>(b32, "b32.txt");
        nnet::load_weights_from_txt<weight36_t, 589824>(w36, "w36.txt");
        nnet::load_weights_from_txt<bias36_t, 256>(b36, "b36.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(s38, "s38.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(b38, "b38.txt");
        nnet::load_weights_from_txt<weight40_t, 65536>(w40, "w40.txt");
        nnet::load_weights_from_txt<bias40_t, 256>(b40, "b40.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(s42, "s42.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(b42, "b42.txt");
        nnet::load_weights_from_txt<weight44_t, 256>(w44, "w44.txt");
        nnet::load_weights_from_txt<bias44_t, 1>(b44, "b44.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

std::cout<<"em_barrel"<<std::endl;
std::cout<<"up_sampling2d"<<std::endl;
    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=3080
    nnet::resize_nearest_me<input_t, config2>(em_barrel, layer2_out); // up_sampling2d

std::cout<<"batch_normalization_5"<<std::endl;
    hls::stream<layer3_t> layer3_out("layer3_out");
    #pragma HLS STREAM variable=layer3_out depth=3080
    nnet::normalize_me<layer2_t, layer3_t, config3>(layer2_out, layer3_out, s3, b3); // batch_normalization_5

std::cout<<"zp2d_conv2d"<<std::endl;
    hls::stream<layer47_t> layer47_out("layer47_out");
    #pragma HLS STREAM variable=layer47_out depth=3540
    nnet::zeropad2d_cl_me<layer3_t, layer47_t, config47>(layer3_out, layer47_out); // zp2d_conv2d

std::cout<<"conv2d"<<std::endl;
    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=3080
    nnet::conv_2d_cl_me<layer47_t, layer4_t, config4>(layer47_out, layer4_out, w4, b4); // conv2d

std::cout<<"leaky_re_lu_5"<<std::endl;
    hls::stream<layer6_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=3080
    nnet::leaky_relu_me<layer4_t, layer6_t, LeakyReLU_config6>(layer4_out, 0.30000001192092896, layer6_out); // leaky_re_lu_5

std::cout<<"max_pooling2d"<<std::endl;
    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=756
    nnet::pooling2d_large_cl_nopad_pad_me<layer6_t, layer7_t, config7>(layer6_out, layer7_out); // max_pooling2d

std::cout<<"zp2d_conv2d_1"<<std::endl;
    hls::stream<layer48_t> layer48_out("layer48_out");
    #pragma HLS STREAM variable=layer48_out depth=870
    nnet::zeropad2d_cl_me<layer7_t, layer48_t, config48>(layer7_out, layer48_out); // zp2d_conv2d_1

std::cout<<"conv2d_1"<<std::endl;
    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=756
    nnet::conv_2d_cl_me<layer48_t, layer8_t, config8>(layer48_out, layer8_out, w8, b8); // conv2d_1

std::cout<<"leaky_re_lu_6"<<std::endl;
    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=756
    nnet::leaky_relu_me<layer8_t, layer10_t, LeakyReLU_config10>(layer8_out, 0.30000001192092896, layer10_out); // leaky_re_lu_6

std::cout<<"zp2d_conv2d_2"<<std::endl;
    hls::stream<layer49_t> layer49_out("layer49_out");
    #pragma HLS STREAM variable=layer49_out depth=870
    nnet::zeropad2d_cl_me<layer10_t, layer49_t, config49>(layer10_out, layer49_out); // zp2d_conv2d_2

std::cout<<"conv2d_2"<<std::endl;
    hls::stream<layer11_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=756
    nnet::conv_2d_cl_me<layer49_t, layer11_t, config11>(layer49_out, layer11_out, w11, b11); // conv2d_2

std::cout<<"leaky_re_lu_7"<<std::endl;
    hls::stream<layer13_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=756
    nnet::leaky_relu_me<layer11_t, layer13_t, LeakyReLU_config13>(layer11_out, 0.30000001192092896, layer13_out); // leaky_re_lu_7

std::cout<<"max_pooling2d_1"<<std::endl;
    hls::stream<layer14_t> layer14_out("layer14_out");
    #pragma HLS STREAM variable=layer14_out depth=182
    nnet::pooling2d_large_cl_nopad_pad_me<layer13_t, layer14_t, config14>(layer13_out, layer14_out); // max_pooling2d_1

std::cout<<"zp2d_conv2d_3"<<std::endl;
    hls::stream<layer50_t> layer50_out("layer50_out");
    #pragma HLS STREAM variable=layer50_out depth=240
    nnet::zeropad2d_cl_me<layer14_t, layer50_t, config50>(layer14_out, layer50_out); // zp2d_conv2d_3

std::cout<<"conv2d_3"<<std::endl;
    hls::stream<layer15_t> layer15_out("layer15_out");
    #pragma HLS STREAM variable=layer15_out depth=182
    nnet::conv_2d_cl_me<layer50_t, layer15_t, config15>(layer50_out, layer15_out, w15, b15); // conv2d_3

std::cout<<"leaky_re_lu_8"<<std::endl;
    hls::stream<layer17_t> layer17_out("layer17_out");
    #pragma HLS STREAM variable=layer17_out depth=182
    nnet::leaky_relu_me<layer15_t, layer17_t, LeakyReLU_config17>(layer15_out, 0.30000001192092896, layer17_out); // leaky_re_lu_8

std::cout<<"zp2d_conv2d_4"<<std::endl;
    hls::stream<layer51_t> layer51_out("layer51_out");
    #pragma HLS STREAM variable=layer51_out depth=240
    nnet::zeropad2d_cl_me<layer17_t, layer51_t, config51>(layer17_out, layer51_out); // zp2d_conv2d_4

std::cout<<"conv2d_4"<<std::endl;
    hls::stream<layer18_t> layer18_out("layer18_out");
    #pragma HLS STREAM variable=layer18_out depth=182
    nnet::conv_2d_cl_me<layer51_t, layer18_t, config18>(layer51_out, layer18_out, w18, b18); // conv2d_4

std::cout<<"leaky_re_lu_9"<<std::endl;
    hls::stream<layer20_t> layer20_out("layer20_out");
    #pragma HLS STREAM variable=layer20_out depth=182
    nnet::leaky_relu_me<layer18_t, layer20_t, LeakyReLU_config20>(layer18_out, 0.30000001192092896, layer20_out); // leaky_re_lu_9

std::cout<<"max_pooling2d_2"<<std::endl;
    hls::stream<layer21_t> layer21_out("layer21_out");
    #pragma HLS STREAM variable=layer21_out depth=42
    nnet::pooling2d_large_cl_nopad_pad_me<layer20_t, layer21_t, config21>(layer20_out, layer21_out); // max_pooling2d_2

std::cout<<"zp2d_conv2d_5"<<std::endl;
    hls::stream<layer52_t> layer52_out("layer52_out");
    #pragma HLS STREAM variable=layer52_out depth=72
    nnet::zeropad2d_cl_me<layer21_t, layer52_t, config52>(layer21_out, layer52_out); // zp2d_conv2d_5

std::cout<<"conv2d_5"<<std::endl;
    hls::stream<layer22_t> layer22_out("layer22_out");
    #pragma HLS STREAM variable=layer22_out depth=42
    nnet::conv_2d_cl_me<layer52_t, layer22_t, config22>(layer52_out, layer22_out, w22, b22); // conv2d_5

std::cout<<"leaky_re_lu_10"<<std::endl;
    hls::stream<layer24_t> layer24_out("layer24_out");
    #pragma HLS STREAM variable=layer24_out depth=42
    nnet::leaky_relu_me<layer22_t, layer24_t, LeakyReLU_config24>(layer22_out, 0.30000001192092896, layer24_out); // leaky_re_lu_10

std::cout<<"zp2d_conv2d_6"<<std::endl;
    hls::stream<layer53_t> layer53_out("layer53_out");
    #pragma HLS STREAM variable=layer53_out depth=72
    nnet::zeropad2d_cl_me<layer24_t, layer53_t, config53>(layer24_out, layer53_out); // zp2d_conv2d_6

std::cout<<"conv2d_6"<<std::endl;
    hls::stream<layer25_t> layer25_out("layer25_out");
    #pragma HLS STREAM variable=layer25_out depth=42
    nnet::conv_2d_cl_me<layer53_t, layer25_t, config25>(layer53_out, layer25_out, w25, b25); // conv2d_6

std::cout<<"leaky_re_lu_11"<<std::endl;
    hls::stream<layer27_t> layer27_out("layer27_out");
    #pragma HLS STREAM variable=layer27_out depth=42
    nnet::leaky_relu_me<layer25_t, layer27_t, LeakyReLU_config27>(layer25_out, 0.30000001192092896, layer27_out); // leaky_re_lu_11

std::cout<<"max_pooling2d_3"<<std::endl;
    hls::stream<layer28_t> layer28_out("layer28_out");
    #pragma HLS STREAM variable=layer28_out depth=9
    nnet::pooling2d_large_cl_nopad_pad_me<layer27_t, layer28_t, config28>(layer27_out, layer28_out); // max_pooling2d_3

std::cout<<"zp2d_conv2d_7"<<std::endl;
    hls::stream<layer54_t> layer54_out("layer54_out");
    #pragma HLS STREAM variable=layer54_out depth=25
    nnet::zeropad2d_cl_me<layer28_t, layer54_t, config54>(layer28_out, layer54_out); // zp2d_conv2d_7

std::cout<<"conv2d_7"<<std::endl;
    hls::stream<layer29_t> layer29_out("layer29_out");
    #pragma HLS STREAM variable=layer29_out depth=9
    nnet::conv_2d_cl_me<layer54_t, layer29_t, config29>(layer54_out, layer29_out, w29, b29); // conv2d_7

std::cout<<"leaky_re_lu_12"<<std::endl;
    hls::stream<layer31_t> layer31_out("layer31_out");
    #pragma HLS STREAM variable=layer31_out depth=9
    nnet::leaky_relu_me<layer29_t, layer31_t, LeakyReLU_config31>(layer29_out, 0.30000001192092896, layer31_out); // leaky_re_lu_12

std::cout<<"zp2d_conv2d_8"<<std::endl;
    hls::stream<layer55_t> layer55_out("layer55_out");
    #pragma HLS STREAM variable=layer55_out depth=25
    nnet::zeropad2d_cl_me<layer31_t, layer55_t, config55>(layer31_out, layer55_out); // zp2d_conv2d_8

std::cout<<"conv2d_8"<<std::endl;
    hls::stream<layer32_t> layer32_out("layer32_out");
    #pragma HLS STREAM variable=layer32_out depth=9
    nnet::conv_2d_cl_me<layer55_t, layer32_t, config32>(layer55_out, layer32_out, w32, b32); // conv2d_8

std::cout<<"leaky_re_lu_13"<<std::endl;
    hls::stream<layer34_t> layer34_out("layer34_out");
    #pragma HLS STREAM variable=layer34_out depth=9
    nnet::leaky_relu_me<layer32_t, layer34_t, LeakyReLU_config34>(layer32_out, 0.30000001192092896, layer34_out); // leaky_re_lu_13

std::cout<<"flatten"<<std::endl;
std::cout<<"dense_6"<<std::endl;
    hls::stream<layer36_t> layer36_out("layer36_out");
    #pragma HLS STREAM variable=layer36_out depth=1
    nnet::dense_ss<layer34_t, layer36_t, config36>(layer34_out, layer36_out, w36, b36); // dense_6

std::cout<<"batch_normalization_15"<<std::endl;
    hls::stream<layer38_t> layer38_out("layer38_out");
    #pragma HLS STREAM variable=layer38_out depth=1
    nnet::normalize_me<layer36_t, layer38_t, config38>(layer36_out, layer38_out, s38, b38); // batch_normalization_15

std::cout<<"leaky_re_lu_14"<<std::endl;
    hls::stream<layer39_t> layer39_out("layer39_out");
    #pragma HLS STREAM variable=layer39_out depth=1
    nnet::leaky_relu_me<layer38_t, layer39_t, LeakyReLU_config39>(layer38_out, 0.30000001192092896, layer39_out); // leaky_re_lu_14

std::cout<<"dense_7"<<std::endl;
    hls::stream<layer40_t> layer40_out("layer40_out");
    #pragma HLS STREAM variable=layer40_out depth=1
    nnet::dense_ss<layer39_t, layer40_t, config40>(layer39_out, layer40_out, w40, b40); // dense_7

std::cout<<"batch_normalization_16"<<std::endl;
    hls::stream<layer42_t> layer42_out("layer42_out");
    #pragma HLS STREAM variable=layer42_out depth=1
    nnet::normalize_me<layer40_t, layer42_t, config42>(layer40_out, layer42_out, s42, b42); // batch_normalization_16

std::cout<<"leaky_re_lu_15"<<std::endl;
    hls::stream<layer43_t> layer43_out("layer43_out");
    #pragma HLS STREAM variable=layer43_out depth=1
    nnet::leaky_relu_me<layer42_t, layer43_t, LeakyReLU_config43>(layer42_out, 0.30000001192092896, layer43_out); // leaky_re_lu_15

std::cout<<"dense_8"<<std::endl;
    hls::stream<layer44_t> layer44_out("layer44_out");
    #pragma HLS STREAM variable=layer44_out depth=1
    nnet::dense_ss<layer43_t, layer44_t, config44>(layer43_out, layer44_out, w44, b44); // dense_8

std::cout<<"activation"<<std::endl;
    nnet::relu_me<layer44_t, result_t, relu_config46>(layer44_out, layer46_out); // activation

}
