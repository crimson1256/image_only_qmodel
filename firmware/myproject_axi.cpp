#include "myproject_axi.h"
    model_default_t s3[4];
    model_default_t b3[4];
    weight4_t w4[1600];
    bias4_t b4[16];
    model_default_t s6[16];
    model_default_t b6[16];
    weight9_t w9[4608];
    bias9_t b9[32];
    model_default_t s11[32];
    model_default_t b11[32];
    weight13_t w13[9216];
    bias13_t b13[32];
    model_default_t s15[32];
    model_default_t b15[32];
    weight18_t w18[18432];
    bias18_t b18[64];
    model_default_t s20[64];
    model_default_t b20[64];
    weight22_t w22[36864];
    bias22_t b22[64];
    model_default_t s24[64];
    model_default_t b24[64];
    weight27_t w27[73728];
    bias27_t b27[128];
    model_default_t s29[128];
    model_default_t b29[128];
    weight31_t w31[147456];
    bias31_t b31[128];
    model_default_t s33[128];
    model_default_t b33[128];
    weight36_t w36[294912];
    bias36_t b36[256];
    model_default_t s38[256];
    model_default_t b38[256];
    weight40_t w40[589824];
    bias40_t b40[256];
    model_default_t s42[256];
    model_default_t b42[256];
    weight45_t w45[589824];
    bias45_t b45[256];
    model_default_t s47[256];
    model_default_t b47[256];
    weight49_t w49[65536];
    bias49_t b49[256];
    model_default_t s51[256];
    model_default_t b51[256];
    weight53_t w53[256];
    bias53_t b53[1];

void myproject_axi(
    input_axi_t in[N_IN],
    output_axi_t out[N_OUT]
        ){

    #pragma HLS INTERFACE axis port=in
    #pragma HLS INTERFACE axis port=out
    #pragma HLS INTERFACE ap_ctrl_none port=return
    #pragma HLS DATAFLOW

    unsigned short in_size = 0;
    unsigned short out_size = 0;

    bool is_last = false;
    hls::stream<input_t> in_local("input_1");
    hls::stream<result_t> out_local("output_1");

    #pragma HLS STREAM variable=in_local depth=N_IN
    #pragma HLS STREAM variable=out_local depth=N_OUT

    for(unsigned i = 0; i < N_IN; ++i) {
        input_t ctype;
            ctype = input_t (in[i].data);
            is_last |= (in[i].last == 1)? true : false;
        in_local.write(ctype);
    }

    myproject(in_local, out_local, s3,b3,w4,b4,s6,b6,w9,b9,s11,b11,w13,b13,s15,b15,w18,b18,s20,b20,w22,b22,s24,b24,w27,b27,s29,b29,w31,b31,s33,b33,w36,b36,s38,b38,w40,b40,s42,b42,w45,b45,s47,b47,w49,b49,s51,b51,w53,b53, in_size, out_size);

    for(unsigned i = 0; i < N_OUT; ++i) {
        result_t ctype = out_local.read();
            bool last = (is_last && (i == N_OUT - 1)) ? true : false;
            out[i] = output_axi_t(ctype, last);
    }
}
