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
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=em_barrel,layer55_out 
    #pragma HLS DATAFLOW 

    const_size_in_1 = N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1;
    const_size_out_1 = N_LAYER_53;

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<model_default_t, 4>(s3, "s3.txt");
        nnet::load_weights_from_txt<model_default_t, 4>(b3, "b3.txt");
        nnet::load_weights_from_txt<weight4_t, 1600>(w4, "w4.txt");
        nnet::load_weights_from_txt<bias4_t, 16>(b4, "b4.txt");
        nnet::load_weights_from_txt<model_default_t, 16>(s6, "s6.txt");
        nnet::load_weights_from_txt<model_default_t, 16>(b6, "b6.txt");
        nnet::load_weights_from_txt<weight9_t, 4608>(w9, "w9.txt");
        nnet::load_weights_from_txt<bias9_t, 32>(b9, "b9.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(s11, "s11.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(b11, "b11.txt");
        nnet::load_weights_from_txt<weight13_t, 9216>(w13, "w13.txt");
        nnet::load_weights_from_txt<bias13_t, 32>(b13, "b13.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(s15, "s15.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(b15, "b15.txt");
        nnet::load_weights_from_txt<weight18_t, 18432>(w18, "w18.txt");
        nnet::load_weights_from_txt<bias18_t, 64>(b18, "b18.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(s20, "s20.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b20, "b20.txt");
        nnet::load_weights_from_txt<weight22_t, 36864>(w22, "w22.txt");
        nnet::load_weights_from_txt<bias22_t, 64>(b22, "b22.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(s24, "s24.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b24, "b24.txt");
        nnet::load_weights_from_txt<weight27_t, 73728>(w27, "w27.txt");
        nnet::load_weights_from_txt<bias27_t, 128>(b27, "b27.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(s29, "s29.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(b29, "b29.txt");
        nnet::load_weights_from_txt<weight31_t, 147456>(w31, "w31.txt");
        nnet::load_weights_from_txt<bias31_t, 128>(b31, "b31.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(s33, "s33.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(b33, "b33.txt");
        nnet::load_weights_from_txt<weight36_t, 294912>(w36, "w36.txt");
        nnet::load_weights_from_txt<bias36_t, 256>(b36, "b36.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(s38, "s38.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(b38, "b38.txt");
        nnet::load_weights_from_txt<weight40_t, 589824>(w40, "w40.txt");
        nnet::load_weights_from_txt<bias40_t, 256>(b40, "b40.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(s42, "s42.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(b42, "b42.txt");
        nnet::load_weights_from_txt<weight45_t, 589824>(w45, "w45.txt");
        nnet::load_weights_from_txt<bias45_t, 256>(b45, "b45.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(s47, "s47.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(b47, "b47.txt");
        nnet::load_weights_from_txt<weight49_t, 65536>(w49, "w49.txt");
        nnet::load_weights_from_txt<bias49_t, 256>(b49, "b49.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(s51, "s51.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(b51, "b51.txt");
        nnet::load_weights_from_txt<weight53_t, 256>(w53, "w53.txt");
        nnet::load_weights_from_txt<bias53_t, 1>(b53, "b53.txt");
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
    hls::stream<layer56_t> layer56_out("layer56_out");
    #pragma HLS STREAM variable=layer56_out depth=3540
    nnet::zeropad2d_cl_me<layer3_t, layer56_t, config56>(layer3_out, layer56_out); // zp2d_conv2d

std::cout<<"conv2d"<<std::endl;
    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=3080
    nnet::conv_2d_cl_me<layer56_t, layer4_t, config4>(layer56_out, layer4_out, w4, b4); // conv2d

std::cout<<"batch_normalization_6"<<std::endl;
    hls::stream<layer6_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=3080
    nnet::normalize_me<layer4_t, layer6_t, config6>(layer4_out, layer6_out, s6, b6); // batch_normalization_6

std::cout<<"leaky_re_lu_5"<<std::endl;
    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=3080
    nnet::leaky_relu_me<layer6_t, layer7_t, LeakyReLU_config7>(layer6_out, 0.30000001192092896, layer7_out); // leaky_re_lu_5

std::cout<<"max_pooling2d"<<std::endl;
    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=756
    nnet::pooling2d_large_cl_nopad_pad_me<layer7_t, layer8_t, config8>(layer7_out, layer8_out); // max_pooling2d

std::cout<<"zp2d_conv2d_1"<<std::endl;
    hls::stream<layer57_t> layer57_out("layer57_out");
    #pragma HLS STREAM variable=layer57_out depth=870
    nnet::zeropad2d_cl_me<layer8_t, layer57_t, config57>(layer8_out, layer57_out); // zp2d_conv2d_1

std::cout<<"conv2d_1"<<std::endl;
    hls::stream<layer9_t> layer9_out("layer9_out");
    #pragma HLS STREAM variable=layer9_out depth=756
    nnet::conv_2d_cl_me<layer57_t, layer9_t, config9>(layer57_out, layer9_out, w9, b9); // conv2d_1

std::cout<<"batch_normalization_7"<<std::endl;
    hls::stream<layer11_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=756
    nnet::normalize_me<layer9_t, layer11_t, config11>(layer9_out, layer11_out, s11, b11); // batch_normalization_7

std::cout<<"leaky_re_lu_6"<<std::endl;
    hls::stream<layer12_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=756
    nnet::leaky_relu_me<layer11_t, layer12_t, LeakyReLU_config12>(layer11_out, 0.30000001192092896, layer12_out); // leaky_re_lu_6

std::cout<<"zp2d_conv2d_2"<<std::endl;
    hls::stream<layer58_t> layer58_out("layer58_out");
    #pragma HLS STREAM variable=layer58_out depth=870
    nnet::zeropad2d_cl_me<layer12_t, layer58_t, config58>(layer12_out, layer58_out); // zp2d_conv2d_2

std::cout<<"conv2d_2"<<std::endl;
    hls::stream<layer13_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=756
    nnet::conv_2d_cl_me<layer58_t, layer13_t, config13>(layer58_out, layer13_out, w13, b13); // conv2d_2

std::cout<<"batch_normalization_8"<<std::endl;
    hls::stream<layer15_t> layer15_out("layer15_out");
    #pragma HLS STREAM variable=layer15_out depth=756
    nnet::normalize_me<layer13_t, layer15_t, config15>(layer13_out, layer15_out, s15, b15); // batch_normalization_8

std::cout<<"leaky_re_lu_7"<<std::endl;
    hls::stream<layer16_t> layer16_out("layer16_out");
    #pragma HLS STREAM variable=layer16_out depth=756
    nnet::leaky_relu_me<layer15_t, layer16_t, LeakyReLU_config16>(layer15_out, 0.30000001192092896, layer16_out); // leaky_re_lu_7

std::cout<<"max_pooling2d_1"<<std::endl;
    hls::stream<layer17_t> layer17_out("layer17_out");
    #pragma HLS STREAM variable=layer17_out depth=182
    nnet::pooling2d_large_cl_nopad_pad_me<layer16_t, layer17_t, config17>(layer16_out, layer17_out); // max_pooling2d_1

std::cout<<"zp2d_conv2d_3"<<std::endl;
    hls::stream<layer59_t> layer59_out("layer59_out");
    #pragma HLS STREAM variable=layer59_out depth=240
    nnet::zeropad2d_cl_me<layer17_t, layer59_t, config59>(layer17_out, layer59_out); // zp2d_conv2d_3

std::cout<<"conv2d_3"<<std::endl;
    hls::stream<layer18_t> layer18_out("layer18_out");
    #pragma HLS STREAM variable=layer18_out depth=182
    nnet::conv_2d_cl_me<layer59_t, layer18_t, config18>(layer59_out, layer18_out, w18, b18); // conv2d_3

std::cout<<"batch_normalization_9"<<std::endl;
    hls::stream<layer20_t> layer20_out("layer20_out");
    #pragma HLS STREAM variable=layer20_out depth=182
    nnet::normalize_me<layer18_t, layer20_t, config20>(layer18_out, layer20_out, s20, b20); // batch_normalization_9

std::cout<<"leaky_re_lu_8"<<std::endl;
    hls::stream<layer21_t> layer21_out("layer21_out");
    #pragma HLS STREAM variable=layer21_out depth=182
    nnet::leaky_relu_me<layer20_t, layer21_t, LeakyReLU_config21>(layer20_out, 0.30000001192092896, layer21_out); // leaky_re_lu_8

std::cout<<"zp2d_conv2d_4"<<std::endl;
    hls::stream<layer60_t> layer60_out("layer60_out");
    #pragma HLS STREAM variable=layer60_out depth=240
    nnet::zeropad2d_cl_me<layer21_t, layer60_t, config60>(layer21_out, layer60_out); // zp2d_conv2d_4

std::cout<<"conv2d_4"<<std::endl;
    hls::stream<layer22_t> layer22_out("layer22_out");
    #pragma HLS STREAM variable=layer22_out depth=182
    nnet::conv_2d_cl_me<layer60_t, layer22_t, config22>(layer60_out, layer22_out, w22, b22); // conv2d_4

std::cout<<"batch_normalization_10"<<std::endl;
    hls::stream<layer24_t> layer24_out("layer24_out");
    #pragma HLS STREAM variable=layer24_out depth=182
    nnet::normalize_me<layer22_t, layer24_t, config24>(layer22_out, layer24_out, s24, b24); // batch_normalization_10

std::cout<<"leaky_re_lu_9"<<std::endl;
    hls::stream<layer25_t> layer25_out("layer25_out");
    #pragma HLS STREAM variable=layer25_out depth=182
    nnet::leaky_relu_me<layer24_t, layer25_t, LeakyReLU_config25>(layer24_out, 0.30000001192092896, layer25_out); // leaky_re_lu_9

std::cout<<"max_pooling2d_2"<<std::endl;
    hls::stream<layer26_t> layer26_out("layer26_out");
    #pragma HLS STREAM variable=layer26_out depth=42
    nnet::pooling2d_large_cl_nopad_pad_me<layer25_t, layer26_t, config26>(layer25_out, layer26_out); // max_pooling2d_2

std::cout<<"zp2d_conv2d_5"<<std::endl;
    hls::stream<layer61_t> layer61_out("layer61_out");
    #pragma HLS STREAM variable=layer61_out depth=72
    nnet::zeropad2d_cl_me<layer26_t, layer61_t, config61>(layer26_out, layer61_out); // zp2d_conv2d_5

std::cout<<"conv2d_5"<<std::endl;
    hls::stream<layer27_t> layer27_out("layer27_out");
    #pragma HLS STREAM variable=layer27_out depth=42
    nnet::conv_2d_cl_me<layer61_t, layer27_t, config27>(layer61_out, layer27_out, w27, b27); // conv2d_5

std::cout<<"batch_normalization_11"<<std::endl;
    hls::stream<layer29_t> layer29_out("layer29_out");
    #pragma HLS STREAM variable=layer29_out depth=42
    nnet::normalize_me<layer27_t, layer29_t, config29>(layer27_out, layer29_out, s29, b29); // batch_normalization_11

std::cout<<"leaky_re_lu_10"<<std::endl;
    hls::stream<layer30_t> layer30_out("layer30_out");
    #pragma HLS STREAM variable=layer30_out depth=42
    nnet::leaky_relu_me<layer29_t, layer30_t, LeakyReLU_config30>(layer29_out, 0.30000001192092896, layer30_out); // leaky_re_lu_10

std::cout<<"zp2d_conv2d_6"<<std::endl;
    hls::stream<layer62_t> layer62_out("layer62_out");
    #pragma HLS STREAM variable=layer62_out depth=72
    nnet::zeropad2d_cl_me<layer30_t, layer62_t, config62>(layer30_out, layer62_out); // zp2d_conv2d_6

std::cout<<"conv2d_6"<<std::endl;
    hls::stream<layer31_t> layer31_out("layer31_out");
    #pragma HLS STREAM variable=layer31_out depth=42
    nnet::conv_2d_cl_me<layer62_t, layer31_t, config31>(layer62_out, layer31_out, w31, b31); // conv2d_6

std::cout<<"batch_normalization_12"<<std::endl;
    hls::stream<layer33_t> layer33_out("layer33_out");
    #pragma HLS STREAM variable=layer33_out depth=42
    nnet::normalize_me<layer31_t, layer33_t, config33>(layer31_out, layer33_out, s33, b33); // batch_normalization_12

std::cout<<"leaky_re_lu_11"<<std::endl;
    hls::stream<layer34_t> layer34_out("layer34_out");
    #pragma HLS STREAM variable=layer34_out depth=42
    nnet::leaky_relu_me<layer33_t, layer34_t, LeakyReLU_config34>(layer33_out, 0.30000001192092896, layer34_out); // leaky_re_lu_11

std::cout<<"max_pooling2d_3"<<std::endl;
    hls::stream<layer35_t> layer35_out("layer35_out");
    #pragma HLS STREAM variable=layer35_out depth=9
    nnet::pooling2d_large_cl_nopad_pad_me<layer34_t, layer35_t, config35>(layer34_out, layer35_out); // max_pooling2d_3

std::cout<<"zp2d_conv2d_7"<<std::endl;
    hls::stream<layer63_t> layer63_out("layer63_out");
    #pragma HLS STREAM variable=layer63_out depth=25
    nnet::zeropad2d_cl_me<layer35_t, layer63_t, config63>(layer35_out, layer63_out); // zp2d_conv2d_7

std::cout<<"conv2d_7"<<std::endl;
    hls::stream<layer36_t> layer36_out("layer36_out");
    #pragma HLS STREAM variable=layer36_out depth=9
    nnet::conv_2d_cl_me<layer63_t, layer36_t, config36>(layer63_out, layer36_out, w36, b36); // conv2d_7

std::cout<<"batch_normalization_13"<<std::endl;
    hls::stream<layer38_t> layer38_out("layer38_out");
    #pragma HLS STREAM variable=layer38_out depth=9
    nnet::normalize_me<layer36_t, layer38_t, config38>(layer36_out, layer38_out, s38, b38); // batch_normalization_13

std::cout<<"leaky_re_lu_12"<<std::endl;
    hls::stream<layer39_t> layer39_out("layer39_out");
    #pragma HLS STREAM variable=layer39_out depth=9
    nnet::leaky_relu_me<layer38_t, layer39_t, LeakyReLU_config39>(layer38_out, 0.30000001192092896, layer39_out); // leaky_re_lu_12

std::cout<<"zp2d_conv2d_8"<<std::endl;
    hls::stream<layer64_t> layer64_out("layer64_out");
    #pragma HLS STREAM variable=layer64_out depth=25
    nnet::zeropad2d_cl_me<layer39_t, layer64_t, config64>(layer39_out, layer64_out); // zp2d_conv2d_8

std::cout<<"conv2d_8"<<std::endl;
    hls::stream<layer40_t> layer40_out("layer40_out");
    #pragma HLS STREAM variable=layer40_out depth=9
    nnet::conv_2d_cl_me<layer64_t, layer40_t, config40>(layer64_out, layer40_out, w40, b40); // conv2d_8

std::cout<<"batch_normalization_14"<<std::endl;
    hls::stream<layer42_t> layer42_out("layer42_out");
    #pragma HLS STREAM variable=layer42_out depth=9
    nnet::normalize_me<layer40_t, layer42_t, config42>(layer40_out, layer42_out, s42, b42); // batch_normalization_14

std::cout<<"leaky_re_lu_13"<<std::endl;
    hls::stream<layer43_t> layer43_out("layer43_out");
    #pragma HLS STREAM variable=layer43_out depth=9
    nnet::leaky_relu_me<layer42_t, layer43_t, LeakyReLU_config43>(layer42_out, 0.30000001192092896, layer43_out); // leaky_re_lu_13

std::cout<<"flatten"<<std::endl;
std::cout<<"dense_6"<<std::endl;
    hls::stream<layer45_t> layer45_out("layer45_out");
    #pragma HLS STREAM variable=layer45_out depth=1
    nnet::dense_ss<layer43_t, layer45_t, config45>(layer43_out, layer45_out, w45, b45); // dense_6

std::cout<<"batch_normalization_15"<<std::endl;
    hls::stream<layer47_t> layer47_out("layer47_out");
    #pragma HLS STREAM variable=layer47_out depth=1
    nnet::normalize_me<layer45_t, layer47_t, config47>(layer45_out, layer47_out, s47, b47); // batch_normalization_15

std::cout<<"leaky_re_lu_14"<<std::endl;
    hls::stream<layer48_t> layer48_out("layer48_out");
    #pragma HLS STREAM variable=layer48_out depth=1
    nnet::leaky_relu_me<layer47_t, layer48_t, LeakyReLU_config48>(layer47_out, 0.30000001192092896, layer48_out); // leaky_re_lu_14

std::cout<<"dense_7"<<std::endl;
    hls::stream<layer49_t> layer49_out("layer49_out");
    #pragma HLS STREAM variable=layer49_out depth=1
    nnet::dense_ss<layer48_t, layer49_t, config49>(layer48_out, layer49_out, w49, b49); // dense_7

std::cout<<"batch_normalization_16"<<std::endl;
    hls::stream<layer51_t> layer51_out("layer51_out");
    #pragma HLS STREAM variable=layer51_out depth=1
    nnet::normalize_me<layer49_t, layer51_t, config51>(layer49_out, layer51_out, s51, b51); // batch_normalization_16

std::cout<<"leaky_re_lu_15"<<std::endl;
    hls::stream<layer52_t> layer52_out("layer52_out");
    #pragma HLS STREAM variable=layer52_out depth=1
    nnet::leaky_relu_me<layer51_t, layer52_t, LeakyReLU_config52>(layer51_out, 0.30000001192092896, layer52_out); // leaky_re_lu_15

std::cout<<"dense_8"<<std::endl;
    hls::stream<layer53_t> layer53_out("layer53_out");
    #pragma HLS STREAM variable=layer53_out depth=1
    nnet::dense_ss<layer52_t, layer53_t, config53>(layer52_out, layer53_out, w53, b53); // dense_8

std::cout<<"activation"<<std::endl;
    nnet::relu_me<layer53_t, result_t, relu_config55>(layer53_out, layer55_out); // activation

}
