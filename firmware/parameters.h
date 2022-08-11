#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_batchnorm.h"
#include "nnet_utils/nnet_batchnorm_stream.h"
#include "nnet_utils/nnet_conv2d.h"
#include "nnet_utils/nnet_conv2d_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_image.h"
#include "nnet_utils/nnet_image_stream.h"
#include "nnet_utils/nnet_padding.h"
#include "nnet_utils/nnet_padding_stream.h"
#include "nnet_utils/nnet_pooling.h"
#include "nnet_utils/nnet_pooling_stream.h"

#include "weights/s3.h"
#include "weights/b3.h"
#include "weights/w4.h"
#include "weights/b4.h"
#include "weights/w8.h"
#include "weights/b8.h"
#include "weights/w11.h"
#include "weights/b11.h"
#include "weights/w15.h"
#include "weights/b15.h"
#include "weights/w18.h"
#include "weights/b18.h"
#include "weights/w22.h"
#include "weights/b22.h"
#include "weights/w25.h"
#include "weights/b25.h"
#include "weights/w29.h"
#include "weights/b29.h"
#include "weights/w32.h"
#include "weights/b32.h"
#include "weights/w36.h"
#include "weights/b36.h"
#include "weights/s38.h"
#include "weights/b38.h"
#include "weights/w40.h"
#include "weights/b40.h"
#include "weights/s42.h"
#include "weights/b42.h"
#include "weights/w44.h"
#include "weights/b44.h"
 
//hls-fpga-machine-learning insert weights

//hls-fpga-machine-learning insert layer-config
// up_sampling2d
struct config2 : nnet::resize_config {
    static const unsigned height = 56;
    static const unsigned width = 11;
    static const unsigned n_chan = 4;
    static const unsigned new_height = 56;
    static const unsigned new_width = 55;
};

// batch_normalization_5
struct config3 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_2*OUT_WIDTH_2*N_CHAN_2;
    static const unsigned n_filt = 4;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// zp2d_conv2d
struct config47 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_2;
    static const unsigned in_width = OUT_WIDTH_2;
    static const unsigned n_chan = N_CHAN_2;
    static const unsigned out_height = OUT_HEIGHT_47;
    static const unsigned out_width = OUT_WIDTH_47;
    static const unsigned pad_top = 2;
    static const unsigned pad_bottom = 2;
    static const unsigned pad_left = 2;
    static const unsigned pad_right = 2;
};

// conv2d
struct config4_mult : nnet::dense_config {
    static const unsigned n_in = 100;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 1600;
    static const unsigned strategy = nnet::resource;
    typedef ap_fixed<16,9> accum_t;
    typedef bias4_t bias_t;
    typedef weight4_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config4 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_47;
    static const unsigned in_width = OUT_WIDTH_47;
    static const unsigned n_chan = N_CHAN_47;
    static const unsigned filt_height = 5;
    static const unsigned filt_width = 5;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_4;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_4;
    static const unsigned out_width = OUT_WIDTH_4;
    static const unsigned reuse_factor = 1600;
    static const unsigned n_zeros = 58;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 9;
    static const unsigned min_width = 9;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,9> accum_t;
    typedef bias4_t bias_t;
    typedef weight4_t weight_t;
    typedef config4_mult mult_config;
};
const ap_uint<config4::filt_height * config4::filt_width> config4::pixels[] = {1,3,7,15,31,30,28,24,16,33,99,231,495,1023,990,924,792,528,1057,3171,7399,15855,32767,31710,29596,25368,16912,33825,101475,236775,507375,1048575,1014750,947100,811800,541200,1082401,3247203,7576807,16236015,33554431,32472030,30307228,25977624,17318416,1082400,3247200,7576800,16236000,33554400,32472000,30307200,25977600,17318400,1082368,3247104,7576576,16235520,33553408,32471040,30306304,25976832,17317888,1081344,3244032,7569408,16220160,33521664,32440320,30277632,25952256,17301504,1048576,3145728,7340032,15728640,32505856,31457280,29360128,25165824,16777216};

// leaky_re_lu_5
struct LeakyReLU_config6 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_4*OUT_WIDTH_4*N_FILT_4;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
    //typedef ap_fixed<32,16> table_t;
};

// max_pooling2d
struct config7 : nnet::pooling2d_config {
    static const unsigned in_height = OUT_HEIGHT_4;
    static const unsigned in_width = OUT_WIDTH_4;
    static const unsigned n_filt = N_FILT_7;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = 2;
    static const unsigned filt_width = 2;
    static const unsigned n_chan = N_FILT_7;

    static const unsigned out_height = OUT_HEIGHT_7;
    static const unsigned out_width = OUT_WIDTH_7;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse = 1;
    typedef ap_fixed<16,9> accum_t;
};

// zp2d_conv2d_1
struct config48 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_7;
    static const unsigned in_width = OUT_WIDTH_7;
    static const unsigned n_chan = N_FILT_7;
    static const unsigned out_height = OUT_HEIGHT_48;
    static const unsigned out_width = OUT_WIDTH_48;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_1
struct config8_mult : nnet::dense_config {
    static const unsigned n_in = 144;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 4608;
    static const unsigned strategy = nnet::resource;
    typedef ap_fixed<16,9> accum_t;
    typedef bias8_t bias_t;
    typedef weight8_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config8 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_48;
    static const unsigned in_width = OUT_WIDTH_48;
    static const unsigned n_chan = N_CHAN_48;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_8;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_8;
    static const unsigned out_width = OUT_WIDTH_8;
    static const unsigned reuse_factor = 4608;
    static const unsigned n_zeros = 168;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,9> accum_t;
    typedef bias8_t bias_t;
    typedef weight8_t weight_t;
    typedef config8_mult mult_config;
};
const ap_uint<config8::filt_height * config8::filt_width> config8::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// leaky_re_lu_6
struct LeakyReLU_config10 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_8*OUT_WIDTH_8*N_FILT_8;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
    //typedef ap_fixed<32,16> table_t;
};

// zp2d_conv2d_2
struct config49 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_8;
    static const unsigned in_width = OUT_WIDTH_8;
    static const unsigned n_chan = N_FILT_8;
    static const unsigned out_height = OUT_HEIGHT_49;
    static const unsigned out_width = OUT_WIDTH_49;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_2
struct config11_mult : nnet::dense_config {
    static const unsigned n_in = 288;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 9216;
    static const unsigned strategy = nnet::resource;
    typedef ap_fixed<16,9> accum_t;
    typedef bias11_t bias_t;
    typedef weight11_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config11 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_49;
    static const unsigned in_width = OUT_WIDTH_49;
    static const unsigned n_chan = N_CHAN_49;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_11;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_11;
    static const unsigned out_width = OUT_WIDTH_11;
    static const unsigned reuse_factor = 9216;
    static const unsigned n_zeros = 429;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,9> accum_t;
    typedef bias11_t bias_t;
    typedef weight11_t weight_t;
    typedef config11_mult mult_config;
};
const ap_uint<config11::filt_height * config11::filt_width> config11::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// leaky_re_lu_7
struct LeakyReLU_config13 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_11*OUT_WIDTH_11*N_FILT_11;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
    //typedef ap_fixed<32,16> table_t;
};

// max_pooling2d_1
struct config14 : nnet::pooling2d_config {
    static const unsigned in_height = OUT_HEIGHT_11;
    static const unsigned in_width = OUT_WIDTH_11;
    static const unsigned n_filt = N_FILT_14;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = 2;
    static const unsigned filt_width = 2;
    static const unsigned n_chan = N_FILT_14;

    static const unsigned out_height = OUT_HEIGHT_14;
    static const unsigned out_width = OUT_WIDTH_14;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse = 1;
    typedef ap_fixed<16,9> accum_t;
};

// zp2d_conv2d_3
struct config50 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_14;
    static const unsigned in_width = OUT_WIDTH_14;
    static const unsigned n_chan = N_FILT_14;
    static const unsigned out_height = OUT_HEIGHT_50;
    static const unsigned out_width = OUT_WIDTH_50;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_3
struct config15_mult : nnet::dense_config {
    static const unsigned n_in = 288;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 18432;
    static const unsigned strategy = nnet::resource;
    typedef ap_fixed<16,9> accum_t;
    typedef bias15_t bias_t;
    typedef weight15_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config15 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_50;
    static const unsigned in_width = OUT_WIDTH_50;
    static const unsigned n_chan = N_CHAN_50;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_15;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_15;
    static const unsigned out_width = OUT_WIDTH_15;
    static const unsigned reuse_factor = 18432;
    static const unsigned n_zeros = 865;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,9> accum_t;
    typedef bias15_t bias_t;
    typedef weight15_t weight_t;
    typedef config15_mult mult_config;
};
const ap_uint<config15::filt_height * config15::filt_width> config15::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// leaky_re_lu_8
struct LeakyReLU_config17 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_15*OUT_WIDTH_15*N_FILT_15;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
    //typedef ap_fixed<32,16> table_t;
};

// zp2d_conv2d_4
struct config51 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_15;
    static const unsigned in_width = OUT_WIDTH_15;
    static const unsigned n_chan = N_FILT_15;
    static const unsigned out_height = OUT_HEIGHT_51;
    static const unsigned out_width = OUT_WIDTH_51;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_4
struct config18_mult : nnet::dense_config {
    static const unsigned n_in = 576;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 36864;
    static const unsigned strategy = nnet::resource;
    typedef ap_fixed<16,9> accum_t;
    typedef bias18_t bias_t;
    typedef weight18_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config18 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_51;
    static const unsigned in_width = OUT_WIDTH_51;
    static const unsigned n_chan = N_CHAN_51;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_18;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_18;
    static const unsigned out_width = OUT_WIDTH_18;
    static const unsigned reuse_factor = 36864;
    static const unsigned n_zeros = 2188;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,9> accum_t;
    typedef bias18_t bias_t;
    typedef weight18_t weight_t;
    typedef config18_mult mult_config;
};
const ap_uint<config18::filt_height * config18::filt_width> config18::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// leaky_re_lu_9
struct LeakyReLU_config20 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_18*OUT_WIDTH_18*N_FILT_18;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
    //typedef ap_fixed<32,16> table_t;
};

// max_pooling2d_2
struct config21 : nnet::pooling2d_config {
    static const unsigned in_height = OUT_HEIGHT_18;
    static const unsigned in_width = OUT_WIDTH_18;
    static const unsigned n_filt = N_FILT_21;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = 2;
    static const unsigned filt_width = 2;
    static const unsigned n_chan = N_FILT_21;

    static const unsigned out_height = OUT_HEIGHT_21;
    static const unsigned out_width = OUT_WIDTH_21;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse = 1;
    typedef ap_fixed<16,9> accum_t;
};

// zp2d_conv2d_5
struct config52 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_21;
    static const unsigned in_width = OUT_WIDTH_21;
    static const unsigned n_chan = N_FILT_21;
    static const unsigned out_height = OUT_HEIGHT_52;
    static const unsigned out_width = OUT_WIDTH_52;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_5
struct config22_mult : nnet::dense_config {
    static const unsigned n_in = 576;
    static const unsigned n_out = 128;
    static const unsigned reuse_factor = 73728;
    static const unsigned strategy = nnet::resource;
    typedef ap_fixed<16,9> accum_t;
    typedef bias22_t bias_t;
    typedef weight22_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config22 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_52;
    static const unsigned in_width = OUT_WIDTH_52;
    static const unsigned n_chan = N_CHAN_52;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_22;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_22;
    static const unsigned out_width = OUT_WIDTH_22;
    static const unsigned reuse_factor = 73728;
    static const unsigned n_zeros = 4496;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,9> accum_t;
    typedef bias22_t bias_t;
    typedef weight22_t weight_t;
    typedef config22_mult mult_config;
};
const ap_uint<config22::filt_height * config22::filt_width> config22::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// leaky_re_lu_10
struct LeakyReLU_config24 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_22*OUT_WIDTH_22*N_FILT_22;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
    //typedef ap_fixed<32,16> table_t;
};

// zp2d_conv2d_6
struct config53 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_22;
    static const unsigned in_width = OUT_WIDTH_22;
    static const unsigned n_chan = N_FILT_22;
    static const unsigned out_height = OUT_HEIGHT_53;
    static const unsigned out_width = OUT_WIDTH_53;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_6
struct config25_mult : nnet::dense_config {
    static const unsigned n_in = 1152;
    static const unsigned n_out = 128;
    static const unsigned reuse_factor = 147456;
    static const unsigned strategy = nnet::resource;
    typedef ap_fixed<16,9> accum_t;
    typedef bias25_t bias_t;
    typedef weight25_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config25 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_53;
    static const unsigned in_width = OUT_WIDTH_53;
    static const unsigned n_chan = N_CHAN_53;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_25;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_25;
    static const unsigned out_width = OUT_WIDTH_25;
    static const unsigned reuse_factor = 147456;
    static const unsigned n_zeros = 11295;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,9> accum_t;
    typedef bias25_t bias_t;
    typedef weight25_t weight_t;
    typedef config25_mult mult_config;
};
const ap_uint<config25::filt_height * config25::filt_width> config25::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// leaky_re_lu_11
struct LeakyReLU_config27 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_25*OUT_WIDTH_25*N_FILT_25;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
    //typedef ap_fixed<32,16> table_t;
};

// max_pooling2d_3
struct config28 : nnet::pooling2d_config {
    static const unsigned in_height = OUT_HEIGHT_25;
    static const unsigned in_width = OUT_WIDTH_25;
    static const unsigned n_filt = N_FILT_28;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = 2;
    static const unsigned filt_width = 2;
    static const unsigned n_chan = N_FILT_28;

    static const unsigned out_height = OUT_HEIGHT_28;
    static const unsigned out_width = OUT_WIDTH_28;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse = 1;
    typedef ap_fixed<16,9> accum_t;
};

// zp2d_conv2d_7
struct config54 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_28;
    static const unsigned in_width = OUT_WIDTH_28;
    static const unsigned n_chan = N_FILT_28;
    static const unsigned out_height = OUT_HEIGHT_54;
    static const unsigned out_width = OUT_WIDTH_54;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_7
struct config29_mult : nnet::dense_config {
    static const unsigned n_in = 1152;
    static const unsigned n_out = 256;
    static const unsigned reuse_factor = 294912;
    static const unsigned strategy = nnet::resource;
    typedef ap_fixed<16,9> accum_t;
    typedef bias29_t bias_t;
    typedef weight29_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config29 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_54;
    static const unsigned in_width = OUT_WIDTH_54;
    static const unsigned n_chan = N_CHAN_54;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_29;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_29;
    static const unsigned out_width = OUT_WIDTH_29;
    static const unsigned reuse_factor = 294912;
    static const unsigned n_zeros = 23906;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,9> accum_t;
    typedef bias29_t bias_t;
    typedef weight29_t weight_t;
    typedef config29_mult mult_config;
};
const ap_uint<config29::filt_height * config29::filt_width> config29::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// leaky_re_lu_12
struct LeakyReLU_config31 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_29*OUT_WIDTH_29*N_FILT_29;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
    //typedef ap_fixed<32,16> table_t;
};

// zp2d_conv2d_8
struct config55 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_29;
    static const unsigned in_width = OUT_WIDTH_29;
    static const unsigned n_chan = N_FILT_29;
    static const unsigned out_height = OUT_HEIGHT_55;
    static const unsigned out_width = OUT_WIDTH_55;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_8
struct config32_mult : nnet::dense_config {
    static const unsigned n_in = 2304;
    static const unsigned n_out = 256;
    static const unsigned reuse_factor = 589824;
    static const unsigned strategy = nnet::resource;
    typedef ap_fixed<16,9> accum_t;
    typedef bias32_t bias_t;
    typedef weight32_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config32 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_55;
    static const unsigned in_width = OUT_WIDTH_55;
    static const unsigned n_chan = N_CHAN_55;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_32;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_32;
    static const unsigned out_width = OUT_WIDTH_32;
    static const unsigned reuse_factor = 589824;
    static const unsigned n_zeros = 63600;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,9> accum_t;
    typedef bias32_t bias_t;
    typedef weight32_t weight_t;
    typedef config32_mult mult_config;
};
const ap_uint<config32::filt_height * config32::filt_width> config32::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// leaky_re_lu_13
struct LeakyReLU_config34 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_32*OUT_WIDTH_32*N_FILT_32;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
    //typedef ap_fixed<32,16> table_t;
};

// dense_6
struct config36 : nnet::dense_config {
    static const unsigned n_in = N_SIZE_1_35;
    static const unsigned n_out = N_LAYER_36;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 256;
    static const unsigned n_zeros = 63602;
    static const unsigned n_nonzeros = 526222;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,9> accum_t;
    typedef bias36_t bias_t;
    typedef weight36_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// batch_normalization_15
struct config38 : nnet::batchnorm_config {
    static const unsigned n_in = N_LAYER_36;
    static const unsigned n_filt = -1;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// leaky_re_lu_14
struct LeakyReLU_config39 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_36;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
    //typedef ap_fixed<32,16> table_t;
};

// dense_7
struct config40 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_36;
    static const unsigned n_out = N_LAYER_40;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 256;
    static const unsigned n_zeros = 3063;
    static const unsigned n_nonzeros = 62473;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,9> accum_t;
    typedef bias40_t bias_t;
    typedef weight40_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// batch_normalization_16
struct config42 : nnet::batchnorm_config {
    static const unsigned n_in = N_LAYER_40;
    static const unsigned n_filt = -1;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// leaky_re_lu_15
struct LeakyReLU_config43 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_40;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
    //typedef ap_fixed<32,16> table_t;
};

// dense_8
struct config44 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_40;
    static const unsigned n_out = N_LAYER_44;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 256;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,9> accum_t;
    typedef bias44_t bias_t;
    typedef weight44_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// activation
struct relu_config46 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_44;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
    //typedef ap_fixed<32,16> table_t;
};


#endif
