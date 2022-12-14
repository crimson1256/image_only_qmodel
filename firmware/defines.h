#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 56
#define N_INPUT_2_1 11
#define N_INPUT_3_1 4
#define OUT_HEIGHT_2 56
#define OUT_WIDTH_2 55
#define N_CHAN_2 4
#define OUT_HEIGHT_56 60
#define OUT_WIDTH_56 59
#define N_CHAN_56 4
#define OUT_HEIGHT_4 56
#define OUT_WIDTH_4 55
#define N_FILT_4 16
#define OUT_HEIGHT_8 28
#define OUT_WIDTH_8 27
#define N_FILT_8 16
#define OUT_HEIGHT_57 30
#define OUT_WIDTH_57 29
#define N_CHAN_57 16
#define OUT_HEIGHT_9 28
#define OUT_WIDTH_9 27
#define N_FILT_9 32
#define OUT_HEIGHT_58 30
#define OUT_WIDTH_58 29
#define N_CHAN_58 32
#define OUT_HEIGHT_13 28
#define OUT_WIDTH_13 27
#define N_FILT_13 32
#define OUT_HEIGHT_17 14
#define OUT_WIDTH_17 13
#define N_FILT_17 32
#define OUT_HEIGHT_59 16
#define OUT_WIDTH_59 15
#define N_CHAN_59 32
#define OUT_HEIGHT_18 14
#define OUT_WIDTH_18 13
#define N_FILT_18 64
#define OUT_HEIGHT_60 16
#define OUT_WIDTH_60 15
#define N_CHAN_60 64
#define OUT_HEIGHT_22 14
#define OUT_WIDTH_22 13
#define N_FILT_22 64
#define OUT_HEIGHT_26 7
#define OUT_WIDTH_26 6
#define N_FILT_26 64
#define OUT_HEIGHT_61 9
#define OUT_WIDTH_61 8
#define N_CHAN_61 64
#define OUT_HEIGHT_27 7
#define OUT_WIDTH_27 6
#define N_FILT_27 128
#define OUT_HEIGHT_62 9
#define OUT_WIDTH_62 8
#define N_CHAN_62 128
#define OUT_HEIGHT_31 7
#define OUT_WIDTH_31 6
#define N_FILT_31 128
#define OUT_HEIGHT_35 3
#define OUT_WIDTH_35 3
#define N_FILT_35 128
#define OUT_HEIGHT_63 5
#define OUT_WIDTH_63 5
#define N_CHAN_63 128
#define OUT_HEIGHT_36 3
#define OUT_WIDTH_36 3
#define N_FILT_36 256
#define OUT_HEIGHT_64 5
#define OUT_WIDTH_64 5
#define N_CHAN_64 256
#define OUT_HEIGHT_40 3
#define OUT_WIDTH_40 3
#define N_FILT_40 256
#define N_SIZE_1_44 2304
#define N_LAYER_45 256
#define N_LAYER_49 256
#define N_LAYER_53 1

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,9> model_default_t;
typedef ap_fixed<16,9> input_t;
typedef ap_fixed<16,9> layer2_t;
typedef ap_fixed<16,9> layer3_t;
typedef ap_fixed<16,9> layer56_t;
typedef ap_fixed<16,9> layer4_t;
typedef ap_fixed<16,9> weight4_t;
typedef ap_fixed<16,9> bias4_t;
typedef ap_fixed<16,9> layer6_t;
typedef ap_fixed<16,9> layer7_t;
typedef ap_fixed<16,9> layer8_t;
typedef ap_fixed<16,9> layer57_t;
typedef ap_fixed<16,9> layer9_t;
typedef ap_fixed<16,9> weight9_t;
typedef ap_fixed<16,9> bias9_t;
typedef ap_fixed<16,9> layer11_t;
typedef ap_fixed<16,9> layer12_t;
typedef ap_fixed<16,9> layer58_t;
typedef ap_fixed<16,9> layer13_t;
typedef ap_fixed<16,9> weight13_t;
typedef ap_fixed<16,9> bias13_t;
typedef ap_fixed<16,9> layer15_t;
typedef ap_fixed<16,9> layer16_t;
typedef ap_fixed<16,9> layer17_t;
typedef ap_fixed<16,9> layer59_t;
typedef ap_fixed<16,9> layer18_t;
typedef ap_fixed<16,9> weight18_t;
typedef ap_fixed<16,9> bias18_t;
typedef ap_fixed<16,9> layer20_t;
typedef ap_fixed<16,9> layer21_t;
typedef ap_fixed<16,9> layer60_t;
typedef ap_fixed<16,9> layer22_t;
typedef ap_fixed<16,9> weight22_t;
typedef ap_fixed<16,9> bias22_t;
typedef ap_fixed<16,9> layer24_t;
typedef ap_fixed<16,9> layer25_t;
typedef ap_fixed<16,9> layer26_t;
typedef ap_fixed<16,9> layer61_t;
typedef ap_fixed<16,9> layer27_t;
typedef ap_fixed<16,9> weight27_t;
typedef ap_fixed<16,9> bias27_t;
typedef ap_fixed<16,9> layer29_t;
typedef ap_fixed<16,9> layer30_t;
typedef ap_fixed<16,9> layer62_t;
typedef ap_fixed<16,9> layer31_t;
typedef ap_fixed<16,9> weight31_t;
typedef ap_fixed<16,9> bias31_t;
typedef ap_fixed<16,9> layer33_t;
typedef ap_fixed<16,9> layer34_t;
typedef ap_fixed<16,9> layer35_t;
typedef ap_fixed<16,9> layer63_t;
typedef ap_fixed<16,9> layer36_t;
typedef ap_fixed<16,9> weight36_t;
typedef ap_fixed<16,9> bias36_t;
typedef ap_fixed<16,9> layer38_t;
typedef ap_fixed<16,9> layer39_t;
typedef ap_fixed<16,9> layer64_t;
typedef ap_fixed<16,9> layer40_t;
typedef ap_fixed<16,9> weight40_t;
typedef ap_fixed<16,9> bias40_t;
typedef ap_fixed<16,9> layer42_t;
typedef ap_fixed<16,9> layer43_t;
typedef ap_fixed<16,9> layer45_t;
typedef ap_fixed<16,9> weight45_t;
typedef ap_uint<1> bias45_t;
typedef ap_fixed<16,9> layer47_t;
typedef ap_fixed<16,9> layer48_t;
typedef ap_fixed<16,9> layer49_t;
typedef ap_fixed<16,9> weight49_t;
typedef ap_uint<1> bias49_t;
typedef ap_fixed<16,9> layer51_t;
typedef ap_fixed<16,9> layer52_t;
typedef ap_fixed<16,9> layer53_t;
typedef ap_fixed<16,9> weight53_t;
typedef ap_fixed<16,9> bias53_t;
typedef ap_fixed<16,9> result_t;

#endif
