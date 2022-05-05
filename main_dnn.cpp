#pragma GCC optimize(0)

#include <iostream>
#include <ap_fixed.h>
#include <ap_int.h>
#include <cmath>
#include <chrono>
#include <stdlib.h>
using namespace std;

#define TEST_2D

typedef unsigned long long uint64;
typedef unsigned long long uint40;
typedef unsigned long long uint32;
typedef unsigned long long uint27;
typedef unsigned long long uint20;
typedef unsigned long long uint18;
typedef unsigned long long uint10;
typedef unsigned long long uint8;
typedef unsigned long long uint4;

uint64_t nanos()
{
    uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::
                  now().time_since_epoch()).count();
    return ns; 
}

template<int F_DIM, int O_DIM, int K_DIM>
void general_convolution(int* feature, int kernel[K_DIM], int* output){
    for (int col = 0; col < O_DIM; col ++){
        int acc = 0;
        for (int k_col = 0; k_col < K_DIM; k_col ++){
            acc += (unsigned long long)feature[col + k_col]*(unsigned long long)kernel[k_col];
        }
        output[col] = acc;
    }
}

template<int CH_IN, int CH_OUT, int H_IN, int W_IN, int H_OUT, int W_OUT>
void general_conv2d(int feature[CH_IN][H_IN][W_IN], int kernel[CH_OUT][CH_IN][3][3], int output[CH_OUT][H_OUT][W_OUT]){
    for (int c_out = 0; c_out < CH_OUT; c_out ++){
        for (int c_in = 0; c_in < CH_IN; c_in ++){
            for (int k_row = 0; k_row < 3; k_row ++){
                for (int row = 0; row < H_OUT; row ++){
                    for (int col = 0; col < W_OUT; col ++){
                        for (int k_col = 0; k_col < 3; k_col ++){
                            unsigned long long partial = (unsigned long long)feature[c_in][row + k_row][col + k_col]*(unsigned long long)kernel[c_out][c_in][k_row][k_col];
                            output[c_out][row][col] += partial;
                        }
                    }
                }
            }
        }
    }
}

void print_bin(uint64 n, int S) {
    cout << n << ": ";
    int binaryNum[64];

    int i = 0;
    while (n > 0) {
        binaryNum[i] = n % 2;
        n = n/2;
        i ++;
    }
 
    for (int j = i - 1; j >= 0; j--) {
        cout << binaryNum[j];
        if (j%S == 0) cout << " ";
    }
    cout << endl;
}

template<int CH_IN, int CH_OUT, int H_IN, int W_IN, int H_OUT, int W_OUT>
void split_conv2d_32bit_unsigned(int feature[CH_IN][H_IN][W_IN], uint32 kernel[CH_OUT][CH_IN][3], int output[CH_OUT][H_OUT][W_OUT]){

    uint64 feature_012 = 0;
    uint64 feature_345 = 0;
    uint64 result_prev = 0;
    uint64 result_this = 0;
    uint64 result = 0;
    uint64 kernel_temp = 0;

    for (int c_out = 0; c_out < CH_OUT; c_out ++){
        for (int c_in = 0; c_in < CH_IN; c_in ++){
            for (int k_row = 0; k_row < 3; k_row ++){
                kernel_temp = (uint64)kernel[c_out][c_in][k_row];
                for (int row = 0; row < H_OUT; row ++){
                    feature_012 = ((uint32)feature[c_in][row + k_row][2] << 20) | ((uint32)feature[c_in][row + k_row][1] << 10) | (uint32)feature[c_in][row + k_row][0];
                    result_prev = (uint64)kernel_temp*(uint64)feature_012;
                    uint32 output_temp[W_OUT] = {};
                    for (int col = 3; col < W_OUT + 3; col += 3){
        
                        feature_345 = ((uint32)feature[c_in][row + k_row][col + 2] << 20) | ((uint32)feature[c_in][row + k_row][col + 1] << 10) | (uint32)feature[c_in][row + k_row][col + 0];
                        
                        result_this = (uint64)kernel_temp*(uint64)feature_345;
                        result = (result_this << 10) + (result_prev >> 20);

                        result_prev = result_this;
                        
                        output_temp[col - 3] =  result & 1023;
                        output_temp[col - 2] = (result >> 10) & 1023;
                        output_temp[col - 1] = (result >> 20) & 1023;
                    }
                    for (int col = 0; col < W_OUT; col ++){
                        output[c_out][row][col] += output_temp[col];
                    }
                }
            }
        }
    }
}

void get_nkg(int a, int w, int p, int q, int nkg[3]){

    bool found = false;
    int max_nk = 0;
    int best_n = 0;
    int best_k = 0; 
    int best_g = 0;
    for (int n = 1; n < 10; n ++){
        for (int k = 1; k < 10; k ++){
            for (int g = 0; g < 6; g ++){
                int s = p + q + g;
                if ((p+(n-1)*s <= a) && (q+(k-1)*s <= w && g >= log2(min(n, k)))) {
                    if (n*k > best_n*best_k){
                        best_n = n;
                        best_k = k;
                        best_g = g;
                    }
                }
            }
        }
    }
    nkg[0] = best_n;
    nkg[1] = best_k;
    nkg[2] = best_g;
}

template<int CH_IN, int CH_OUT, int H_IN, int W_IN, int H_OUT, int W_OUT>
void conv2d(int feature[CH_IN][H_IN][W_IN], int kernel[CH_OUT][CH_IN][3][3], int output[CH_OUT][H_OUT][W_OUT], bool hikonv){
    if (hikonv) {
        uint32 kernel3d_012[CH_OUT][CH_IN][3] = {};
        for (int i = 0; i < CH_OUT; i ++){
            for (int j = 0; j < CH_IN; j ++){
                for (int k = 0; k < 3; k ++){
                    uint32 kernel3d_012_temp = 0;
                    kernel3d_012_temp |= (uint32)kernel[i][j][k][0] << 20;
                    kernel3d_012_temp |= (uint32)kernel[i][j][k][1] << 10;
                    kernel3d_012_temp |= (uint32)kernel[i][j][k][2];
                    kernel3d_012[i][j][k] = kernel3d_012_temp;
                }
            }
        }
        // uint10 output3d_dsp[CH_OUT][H_OUT][W_OUT] = {};
        split_conv2d_32bit_unsigned<CH_IN, CH_OUT, H_IN, W_IN, H_OUT, W_OUT>(feature, kernel3d_012, output);
    } else {
        general_conv2d<CH_IN, CH_OUT, H_IN, W_IN, H_OUT, W_OUT>(feature, kernel, output);
    }
}

template<int CH, int H, int W>
void relu(int feature_in[CH][H][W], int feature_out[CH][H][W]) {
    for (int i = 0; i < CH; i ++) {
        for (int j = 0; j < H; j ++) {
            for (int k = 0; k < W; k ++) {
                if (feature_in[i][j][k] > 0)
                    feature_out[i][j][k] = feature_in[i][j][k];
                else
                    feature_out[i][j][k] = 0;
            }
        }
    }
}

template<int CH, int H, int W>
void maxpool(int feature_in[CH][H][W], int feature_out[CH][H][W]) {
    for (int i = 0; i < CH; i ++) {
        for (int j = 0; j < H/2; j ++) {
            for (int k = 0; k < W/2; k ++) {
                float max = 0;
                for (int jj = 0; jj < 2; jj ++) {
                    for (int kk = 0; kk < 2; kk ++) {
                        float curr = feature_in[i][2*j + jj][2*k + kk];
                        if (curr > max) max = curr;
                    }
                }
                feature_out[i][j][k] = max;
            }
        }
    }
}

template<int CH_IN, int CH_OUT, int H_IN, int W_IN, int H_OUT, int W_OUT>
void bundle(int feature_in[1][10][20], int feature_out[1][10][20], bool hikonv) {
    int kernel[CH_OUT][CH_IN][3][3] = {};
    for (int i = 0; i < CH_OUT; i ++){
        for (int j = 0; j < CH_IN; j ++){
            for (int k = 0; k < 3; k ++){
                for (int l = 0; l < 3; l ++){
                    kernel[i][j][k][l] = rand() %2;
                }
            }
        }
    }
    for (int ch_out_t = 0; ch_out_t < CH_OUT; ch_out_t ++) {
        for (int ch_in_t = 0; ch_in_t < CH_IN; ch_in_t ++) {
            int kernel_slice[1][1][3][3] = {};
            for (int k = 0; k < 3; k ++){
                for (int l = 0; l < 3; l ++){
                    kernel_slice[1][1][k][l] = kernel[ch_out_t][ch_in_t][k][l];
                }
            }
            for (int h_t = 0; h_t < H_IN/10; h_t ++) {
                for (int w_t = 0; w_t < W_IN/20; w_t ++) {
                    int output[1][10][20] = {};
                    conv2d<1, 1, 10, 20, 10, 20>(feature_in, kernel_slice, output, hikonv);
                    relu<1, 10, 20>(output, feature_out);
                }
            }
        }
    }
}

template<int CH_IN, int CH_OUT, int H_IN, int W_IN, int H_OUT, int W_OUT>
void bundle_maxpool(int feature_in[1][10][20], int feature_out[1][10][20], bool hikonv) {
    int kernel[CH_OUT][CH_IN][3][3] = {};
    for (int i = 0; i < CH_OUT; i ++){
        for (int j = 0; j < CH_IN; j ++){
            for (int k = 0; k < 3; k ++){
                for (int l = 0; l < 3; l ++){
                    kernel[i][j][k][l] = rand() %2;
                }
            }
        }
    }
    for (int ch_out_t = 0; ch_out_t < CH_OUT; ch_out_t ++) {
        for (int ch_in_t = 0; ch_in_t < CH_IN; ch_in_t ++) {
            int kernel_slice[1][1][3][3] = {};
            for (int k = 0; k < 3; k ++){
                for (int l = 0; l < 3; l ++){
                    kernel_slice[1][1][k][l] = kernel[ch_out_t][ch_in_t][k][l];
                }
            }
            for (int h_t = 0; h_t < H_IN/10; h_t ++) {
                for (int w_t = 0; w_t < W_IN/20; w_t ++) {
                    int output[1][10][20] = {};
                    int output_actv[1][10][20] = {};
                    conv2d<1, 1, 10, 20, 10, 20>(feature_in, kernel_slice, output, hikonv);
                    relu<1, 10, 20>(output, output_actv);
                    maxpool<1, 10, 20>(output_actv, feature_out);
                }
            }
        }
    }
}

void ultranet(int output[36][10][20], bool hikonv) {
    int feature[1][10][20] = {};
    for (int i = 0; i < 1; i ++){
        for (int j = 0; j < 10; j ++){
            for (int k = 0; k < 20; k ++){
                feature[i][j][k] = rand() %7;
            }
        }
    }    
    long long ts = 0;
    long long tt = 0;

    int feature_l0[1][10][20] = {};
    ts = nanos();
    bundle_maxpool<3, 16, 160, 320, 80, 160>(feature, feature_l0, hikonv);
    tt = nanos();
    cout << "Layer 1 Elapsed: " << (tt - ts) << " ns" << endl;

    int feature_l1[1][10][20] = {};
    ts = nanos();
    bundle_maxpool<16, 32, 80, 160, 40, 80>(feature_l0, feature_l1, hikonv);
    tt = nanos();
    cout << "Layer 2 Elapsed: " << (tt - ts) << " ns" << endl;

    int feature_l2[1][10][20] = {};
    ts = nanos();
    bundle_maxpool<32, 64, 40, 80, 20, 40>(feature_l1, feature_l2, hikonv);
    tt = nanos();
    cout << "Layer 3 Elapsed: " << (tt - ts) << " ns" << endl;

    int feature_l3[1][10][20] = {};
    ts = nanos();
    bundle_maxpool<64, 64, 20, 40, 10, 20>(feature_l2, feature_l3, hikonv);
    tt = nanos();
    cout << "Layer 4 Elapsed: " << (tt - ts) << " ns" << endl;

    int feature_l4[1][10][20] = {};
    ts = nanos();
    bundle<64, 64, 10, 20, 10, 20>(feature_l3, feature_l4, hikonv);
    tt = nanos();
    cout << "Layer 5 Elapsed: " << (tt - ts) << " ns" << endl;

    int feature_l5[1][10][20] = {};
    ts = nanos();
    bundle<64, 64, 10, 20, 10, 20>(feature_l4, feature_l5, hikonv);
    tt = nanos();
    cout << "Layer 6 Elapsed: " << (tt - ts) << " ns" << endl;

    int feature_l6[1][10][20] = {};
    ts = nanos();
    bundle<64, 64, 10, 20, 10, 20>(feature_l5, feature_l6, hikonv);
    tt = nanos();
    cout << "Layer 7 Elapsed: " << (tt - ts) << " ns" << endl;

    int feature_l7[1][10][20] = {};
    ts = nanos();
    bundle<64, 64, 10, 20, 10, 20>(feature_l6, feature_l7, hikonv);
    tt = nanos();
    cout << "Layer 8 Elapsed: " << (tt - ts) << " ns" << endl;

    int kernel_l8[36][64][3][3] = {};
    ts = nanos();
    for (int i = 0; i < 36; i ++){
        for (int j = 0; j < 64; j ++){
            for (int k = 0; k < 3; k ++){
                for (int l = 0; l < 3; l ++){
                    kernel_l8[i][j][k][l] = rand() %2;
                }
            }
        }
    }
    conv2d<64, 36, 10, 20, 10, 20>(feature_l7, kernel_l8, output, hikonv);
    tt = nanos();
    cout << "Layer 9 Elapsed: " << (tt - ts) << " ns" << endl;
}

int main(){

    const int a_bits = 32;
    const int w_bits = 32;

    const int reps = 1;
    int count = 0;

    long long ts = 0;
    long long tt = 0;

    int output3d[36][10][20] = {};

    cout << "======================================" << endl;
    
    #ifdef TEST_2D
    cout << "Normal ultranet tests:" << endl;
    ts = nanos();
    for (int i = 0; i < reps; i ++){
        ultranet(output3d, false);
    }
    tt = nanos();

    #ifdef PRINT
    cout << "Output:" << endl;
    for (int i = 0; i < c_out; i ++){
        for (int j = 0; j < h_out; j ++){
            for (int k = 0; k < w_out; k ++){
                cout << output3d[i][j][k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;
    // for (int i = 0; i < o_dim; i ++){
    //     print_bin(output[i]);
    // }    
    #endif

    cout << "Elapsed: " << (tt - ts)/reps << " ns" << endl;
    cout << "======================================" << endl;
    cout << endl << "======================================" << endl;
    #endif


    int output3d_dsp[36][10][20] = {};
    cout << "dsp ultranet tests:" << endl;
    ts = nanos();
    for (int i = 0; i < reps; i++){
        ultranet(output3d_dsp, true);
    }
    tt = nanos();

    #ifdef PRINT
    cout << "Output: " << endl;
    for (int i = 0; i < c_out; i ++){
        for (int j = 0; j < h_out; j ++){
            for (int k = 0; k < w_out; k ++){
                cout << output3d_dsp[i][j][k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    #endif

    cout << "Elapsed: " << (tt - ts)/reps << " ns" << endl;
    // count = 0;
    // for (int i = 0; i < 36; i ++){
    //     for (int j = 0; j < 10; j ++){
    //         for (int k = 0; k < 20; k ++){
    //             if ((int)output3d_dsp[i][j][k] != output3d[i][j][k]) {
    //                 count ++;
    //                 // cout << "MISMATCH @ (i, j, k) = (" << i << ", " << j << ", " << k << ")  " << output3d_dsp[i][j][k] << " vs. " << output3d[i][j][k] << endl;
    //             }
    //         }
    //     }
    // }
    
    // if (count == 0) cout << "ALL TEST PASS" << endl;
    // else cout << count << " TESTS FAIL" << endl;
    cout << "======================================" << endl << endl;

    return 0;
}
