//Numpy array shape [16]
//Min -0.117187500000
//Max 0.054687500000
//Number of zeros 1

#ifndef B4_H_
#define B4_H_

#ifndef __SYNTHESIS__
bias4_t b4[16];
#else
bias4_t b4[16] = {0.0078125, 0.0312500, 0.0390625, -0.0078125, -0.0312500, -0.0703125, 0.0546875, -0.0937500, 0.0078125, -0.0312500, 0.0000000, -0.1171875, -0.0390625, 0.0468750, -0.0546875, -0.0703125};
#endif

#endif