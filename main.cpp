#pragma GCC optimize(0)

#include <iostream>
#include <ap_fixed.h>
#include <ap_int.h>
#include <cmath>
#include <chrono>
#include <stdlib.h>
using namespace std;

// #define PRINT
// #define PRINT_PROC
#define TEST_1D
// #define TEST_2D
// #define TEST_MISC
// #define TEST_UNIT

#define INT_TEST

#ifdef INT_TEST
    typedef unsigned long long uint64;
    typedef unsigned long long uint40;
    typedef unsigned int uint32;
    typedef unsigned int uint27;
    typedef unsigned int uint20;
    typedef unsigned int uint18;
    typedef unsigned int uint10;
    typedef unsigned int uint8;
    typedef unsigned int uint4;
#else
    typedef ap_uint<64> uint64;
    typedef ap_uint<40> uint40;
    typedef ap_uint<32> uint32;
    typedef ap_uint<27> uint27;
    typedef ap_uint<20> uint20;
    typedef ap_uint<18> uint18;
    typedef ap_uint<10> uint10;
    typedef ap_uint<8>  uint8;
    typedef ap_uint<4>  uint4;
#endif

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

template<int f_dim, int k_dim, int o_dim>
long long general_conv1d_test(int reps, int* feature, int kernel[k_dim], int output[o_dim]){
    long ts = nanos();
    for (int i = 0; i < reps; i ++){
        general_convolution<f_dim, o_dim, k_dim>(feature, kernel, output);
    }
    long tt = nanos();

    #ifdef PRINT
    cout << "Output:" << endl;
    for (int i = 0; i < o_dim; i ++){
        cout << output[i] << " ";
    }
    cout << endl;
    #endif
    #ifdef PRINT_PROC
    // cout << endl;
    // for (int i = 0; i < o_dim; i ++){
    //     print_bin("output[i]", output[i], 10);
    // }
    #endif

    return (tt - ts)/reps;
}

void print_bin(string comment, uint64 n, int S) {
    int binaryNum[64];
    uint64 n0 = n;
    int i = 0;
    while (n > 0) {
        binaryNum[i] = n % 2;
        n = n/2;
        i ++;
    }
    
    #ifdef PRINT_PROC
    int cnt = i - 1;
    for (int j = 63; j >= 0; j --) {
        if (j <= i - 1) {
            cout << binaryNum[cnt];
            if (cnt%S == 0) cout << " ";
            cnt --;
        } else {
            cout << " ";
            if (j%S == 0) cout << " ";
        }
    }
    cout << " --" << comment;
    cout << ": " << n0;
    cout << endl;
    #endif
}

int sign(uint64 num) {
    return num >> 63 & 1;
}

template<int F_DIM, int O_DIM>
void split_conv1d_32bit_p8q8_signed(uint4* feature, uint32 kernel, uint10* output){

    uint64 feature_012 = 0;
    uint64 feature_345 = 0;
    uint64 feature_012_correct = 0;
    uint64 feature_345_correct = 0;
    uint64 result_prev = 0;
    uint64 result_this = 0;
    uint64 result = 0;

    feature_012 = feature[0];
    feature_012 += feature[1] << 17;

    result_prev = kernel*feature_012;

    for (int col = 2; col < O_DIM + 2; col += 2){
        feature_345 = feature[col];
        feature_345 += feature[col + 1] << 17;

        result_this = kernel*feature_345;
        result = (result_prev >> 17) + (result_this << 17);

        int sign_2 = (result_prev >> 16) & 1;
        int sign_1 = (result >> 16) & 1;
        output[col - 2] =  (result & 131071) + sign_2;
        output[col - 1] = ((result >> 17) & 131071) + sign_1;

        result_prev = result_this;
    }
}

template<int F_DIM, int O_DIM>
void split_conv1d_32bit_p7q7_signed(uint4* feature, uint32 kernel, uint10* output){

    uint64 feature_012 = 0;
    uint64 feature_345 = 0;
    uint64 result_prev = 0;
    uint64 result_this = 0;
    uint64 result = 0;

    int s = 15;
    int mask = pow(2, s) - 1;
    feature_012 = feature[0];
    feature_012 += feature[1] << 15;

    result_prev = kernel*feature_012;

    for (int col = 2; col < O_DIM + 2; col += 2){

        feature_345 = feature[col];
        feature_345 += feature[col + 1] << 15;

        result_this = kernel*feature_345;
        result = (result_prev >> 15) + (result_this << 15);

        int sign_2 = (result_prev >> 14) & 1;
        int sign_1 = (result >> 14) & 1;
        output[col - 2] =  (result & 32767) + sign_2;
        output[col - 1] = ((result >> 15) & 32767) + sign_1;

        result_prev = result_this;
    }
}


template<int F_DIM, int O_DIM>
void split_conv1d_32bit_p6q6_signed(uint4* feature, uint32 kernel, uint10* output){

    uint64 feature_012 = 0;
    uint64 feature_345 = 0;
    uint64 result_prev = 0;
    uint64 result_this = 0;
    uint64 result = 0;

    feature_012 = feature[0];
    feature_012 += feature[1] << 13;
    feature_012 += feature[2] << 26;
    
    result_prev = kernel*feature_012;

    for (int col = 3; col < O_DIM + 3; col += 3){
        feature_345 = feature[col];
        feature_345 += feature[col + 1] << 13;
        feature_345 += feature[col + 2] << 26;

        result_this = kernel*feature_345;
        uint64 result_prev_shifted = ((0 - ((result_prev >> 63) & 1)) << 38) + (result_prev >> 26);
        result = (result_this << 13) + result_prev_shifted;

        int sign_3 = (result_prev >> 25) & 1;
        int sign_2 = (result >> 12) & 1;
        int sign_1 = (result >> 25) & 1;
        output[col - 3] =  (result & 8191) + sign_3;
        output[col - 2] = ((result >> 13) & 8191) + sign_2;
        output[col - 1] = ((result >> 26) & 8191) + sign_1;

        result_prev = result_this;
    }
}

template<int F_DIM, int O_DIM>
void split_conv1d_32bit_p5q5_signed(uint4* feature, uint32 kernel, uint10* output){

    uint64 feature_012 = 0;
    uint64 feature_345 = 0;
    uint64 result_prev = 0;
    uint64 result_this = 0;
    uint64 result = 0;

    feature_012 = feature[0];
    feature_012 += feature[1] << 12;
    feature_012 += feature[2] << 24;

    result_prev = kernel*feature_012;

    for (int col = 3; col < O_DIM + 3; col += 3){


        feature_345 = feature[col];
        feature_345 += feature[col + 1] << 12;
        feature_345 += feature[col + 2] << 24;

        result_this = kernel*feature_345;
        result = (result_this << 12) + (result_prev >> 24);

        int sign_3 = (result_prev >> 23) & 1;
        int sign_2 = (result >> 11) & 1;
        int sign_1 = (result >> 23) & 1;
        output[col - 3] =  (result & 4095) + sign_3;
        output[col - 2] = ((result >> 12) & 4095) + sign_2;
        output[col - 1] = ((result >> 24) & 4095) + sign_1;

        result_prev = result_this;
    }
}

template<int F_DIM, int O_DIM>
void split_conv1d_32bit_p4q4_signed(uint4* feature, uint32 kernel, uint10* output){

    uint64 feature_012 = 0;
    uint64 feature_345 = 0;
    uint64 result_prev = 0;
    uint64 result_this = 0;
    uint64 result = 0;

    feature_012 = feature[0];
    feature_012 += feature[1] << 10;
    feature_012 += feature[2] << 20;

    result_prev = kernel*feature_012;

    for (int col = 3; col < O_DIM + 3; col += 3){

        feature_345 = feature[col];
        feature_345 += feature[col + 1] << 10;
        feature_345 += feature[col + 2] << 20;

        result_this = kernel*feature_345;
        result = (result_this << 10) + (result_prev >> 20);

        int sign_3 = (result_prev >> 19) & 1;
        int sign_2 = (result >> 9) & 1;
        int sign_1 = (result >> 19) & 1;
        output[col - 3] =  (result & 1023) + sign_3;
        output[col - 2] = ((result >> 10) & 1023) + sign_2;
        output[col - 1] = ((result >> 20) & 1023) + sign_1;

        result_prev = result_this;
    }
}

template<int F_DIM, int O_DIM>
void split_conv1d_32bit_p3q3_signed(uint4* feature, uint32 kernel, uint10* output){

    uint64 feature_012 = 0;
    uint64 feature_345 = 0;
    uint64 result_prev = 0;
    uint64 result_this = 0;
    uint64 result = 0;

    feature_012 = feature[0];
    feature_012 += feature[1] << 8;
    feature_012 += feature[2] << 16;
    feature_012 += feature[3] << 24;

    result_prev = kernel*feature_012;

    for (int col = 4; col < O_DIM + 4; col += 4){

        feature_345 = feature[col];
        feature_345 += feature[col + 1] << 8;
        feature_345 += feature[col + 2] << 16;
        feature_345 += feature[col + 3] << 24;

        result_this = kernel*feature_345;
        result = (result_this << 8) + (result_prev >> 24);

        int sign_4 = (result_prev >> 15) & 1;
        int sign_3 = (result >> 7) & 1;
        int sign_2 = (result >> 15) & 1;
        int sign_1 = (result >> 23) & 1;
        output[col - 4] =  (result & 255) + sign_4;
        output[col - 3] = ((result >> 8) & 255) + sign_3;
        output[col - 2] = ((result >> 16) & 255) + sign_2;
        output[col - 1] = ((result >> 24) & 255) + sign_1;

        result_prev = result_this;
    }
}

template<int F_DIM, int O_DIM>
void split_conv1d_32bit_p2q2_signed(uint4* feature, uint32 kernel, uint10* output){

    uint64 feature_012 = 0;
    uint64 feature_345 = 0;
    uint64 result_prev = 0;
    uint64 result_this = 0;
    uint64 result = 0;

    feature_012 = feature[0];
    feature_012 += feature[1] << 7;
    feature_012 += feature[2] << 14;
    feature_012 += feature[3] << 21;
    feature_012 += feature[4] << 28;

    result_prev = kernel*feature_012;

    for (int col = 5; col < O_DIM + 5; col += 5){

        feature_345 = feature[col];
        feature_345 += feature[col + 1] << 7;
        feature_345 += feature[col + 2] << 14;
        feature_345 += feature[col + 3] << 21;
        feature_345 += feature[col + 4] << 28;

        result_this = kernel*feature_345;
        result = (result_this << 7) + (result_prev >> 28);

        int sign_5 = (result_prev >> 13) & 1;
        int sign_4 = (result >> 6) & 1;
        int sign_3 = (result >> 13) & 1;
        int sign_2 = (result >> 20) & 1;
        int sign_1 = (result >> 27) & 1;
        output[col - 5] =  (result & 127) + sign_5;
        output[col - 4] = ((result >> 7) & 127) + sign_4;
        output[col - 3] = ((result >> 14) & 127) + sign_3;
        output[col - 2] = ((result >> 21) & 127) + sign_2;
        output[col - 1] = ((result >> 28) & 127) + sign_1;

        result_prev = result_this;
    }
}
template<int F_DIM, int O_DIM>
void split_conv1d_32bit_p1q1_signed(uint4* feature, uint32 kernel, uint10* output){

    uint32 feature_012 = 0;
    uint32 feature_345 = 0;
    uint64 result_prev = 0;
    uint64 result_this = 0;
    uint64 result = 0;

    feature_012 = ((uint32)feature[6] << 30) | ((uint32)feature[5] << 25) | ((uint32)feature[4] << 20) | ((uint32)feature[3] << 15) | ((uint32)feature[2] << 10) | ((uint32)feature[1] << 5) | (uint32)feature[0];
    result_prev = (uint64)kernel*(uint64)feature_012;

    for (int col = 7; col < O_DIM + 7; col += 7){
        feature_345 = ((uint32)feature[col + 6] << 30) | ((uint32)feature[col + 5] << 25) | ((uint32)feature[col + 4] << 20) | ((uint32)feature[col + 3] << 15) | ((uint32)feature[col + 2] << 10) | ((uint32)feature[col + 1] << 5) | (uint32)feature[col];
        result_this = (uint64)kernel*(uint64)feature_345;
        result = (result_this << 5) + (result_prev >> 30);

        result_prev = result_this;

        output[col - 7] =  result & 31;
        output[col - 6] = (result >> 5) & 31;
        output[col - 5] = (result >> 10) & 31;
        output[col - 4] = (result >> 15) & 31;
        output[col - 3] = (result >> 20) & 31;
        output[col - 2] = (result >> 25) & 31;
        output[col - 1] = (result >> 30) & 31;
    }
}


template<int F_DIM, int O_DIM>
void split_conv1d_32bit_p8q8(uint4* feature, uint32 kernel, uint10* output){

    uint32 feature_012 = 0;
    uint32 feature_345 = 0;
    uint64 result_prev = 0;
    uint64 result_this = 0;
    uint64 result = 0;

    feature_012 = ((uint32)feature[1] << 17) | (uint32)feature[0];
    result_prev = (uint64)kernel*(uint64)feature_012;

    for (int col = 2; col < O_DIM + 2; col += 2){
        feature_345 = (uint32)feature[col+0]|((uint32)feature[col+1]<<17);
        result_this = (uint64)kernel*(uint64)feature_345;
        result = (result_prev >> 17) + (result_this << 17);

        result_prev = result_this;

        output[col - 2] = result & 131071;
        output[col - 1] = (result >> 17) & 131071;
    }
}

template<int F_DIM, int O_DIM>
void split_conv1d_32bit_p7q7(uint4* feature, uint32 kernel, uint10* output){

    uint32 feature_012 = 0;
    uint32 feature_345 = 0;
    uint64 result_prev = 0;
    uint64 result_this = 0;
    uint64 result = 0;

    feature_012 = ((uint32)feature[1] << 15) | (uint32)feature[0];
    result_prev = (uint64)kernel*(uint64)feature_012;

    for (int col = 2; col < O_DIM + 2; col += 2){
        feature_345 = (uint32)feature[col+0]|((uint32)feature[col+1]<<15);
        result_this = (uint64)kernel*(uint64)feature_345;
        result = (result_prev >> 15) + (result_this << 15);

        result_prev = result_this;

        output[col - 2] = result & 32767;
        output[col - 1] = (result >> 15) & 32767;
    }
}

template<int F_DIM, int O_DIM>
void split_conv1d_32bit_p6q6(uint4* feature, uint32 kernel, uint10* output){

    uint32 feature_012 = 0;
    uint32 feature_345 = 0;
    uint64 result_prev = 0;
    uint64 result_this = 0;
    uint64 result = 0;

    feature_012 = ((uint32)feature[2] << 26) | ((uint32)feature[1] << 13) | (uint32)feature[0];
    result_prev = (uint64)kernel*(uint64)feature_012;

    for (int col = 3; col < O_DIM + 3; col += 3){
        feature_345 = (uint32)feature[col+0]|((uint32)feature[col+1]<<13)|((uint32)feature[col+2]<<26);
        result_this = (uint64)kernel*(uint64)feature_345;
        result = (result_prev >> 26) + (result_this << 13);

        result_prev = result_this;

        output[col - 3] =  result & 8191;
        output[col - 2] = (result >> 13) & 8191;
        output[col - 1] = (result >> 26) & 8191;
    }
}

template<int F_DIM, int O_DIM>
void split_conv1d_32bit_p5q5(uint4* feature, uint32 kernel, uint10* output){

    uint32 feature_012 = 0;
    uint32 feature_345 = 0;
    uint64 result_prev = 0;
    uint64 result_this = 0;
    uint64 result = 0;

    feature_012 = ((uint32)feature[2] << 24) | ((uint32)feature[1] << 12) | (uint32)feature[0];
    result_prev = (uint64)kernel*(uint64)feature_012;

    for (int col = 3; col < O_DIM + 3; col += 3){
        feature_345 = (uint32)feature[col+0]|((uint32)feature[col+1]<<12)|((uint32)feature[col+2]<<24);
        result_this = (uint64)kernel*(uint64)feature_345;
        result = (result_this << 12) + (result_prev >> 24);

        result_prev = result_this;

        output[col - 3] =  result & 4095;
        output[col - 2] = (result >> 12) & 4095;
        output[col - 1] = (result >> 24) & 4095;
    }
}

template<int F_DIM, int O_DIM>
void split_conv1d_32bit_p4q4(uint4* feature, uint32 kernel, uint10* output){

    uint32 feature_012 = 0;
    uint32 feature_345 = 0;
    uint64 result_prev = 0;
    uint64 result_this = 0;
    uint64 result = 0;

    feature_012 = ((uint32)feature[2] << 20) | ((uint32)feature[1] << 10) | (uint32)feature[0];
    result_prev = (uint64)kernel*(uint64)feature_012;

    for (int col = 3; col < O_DIM + 3; col += 3){
        feature_345 = (uint32)feature[col+0]|((uint32)feature[col+1]<<10)|((uint32)feature[col+2]<<20);
        result_this = (uint64)kernel*(uint64)feature_345;
        result = (result_this << 10) + (result_prev >> 20);

        result_prev = result_this;

        output[col - 3] =  result & 1023;
        output[col - 2] = (result >> 10) & 1023;
        output[col - 1] = (result >> 20) & 1023;
    }
}


template<int F_DIM, int O_DIM>
void split_conv1d_32bit_p3q3(uint4* feature, uint32 kernel, uint10* output){

    uint32 feature_012 = 0;
    uint32 feature_345 = 0;
    uint64 result_prev = 0;
    uint64 result_this = 0;
    uint64 result = 0;

    feature_012 = ((uint32)feature[3] << 24) | ((uint32)feature[2] << 16) | ((uint32)feature[1] << 8) | (uint32)feature[0];
    result_prev = (uint64)kernel*(uint64)feature_012;

    for (int col = 4; col < O_DIM + 4; col += 4){
        feature_345 = ((uint32)feature[col + 3] << 24) | ((uint32)feature[col + 2] << 16) | ((uint32)feature[col + 1] << 8) | (uint32)feature[col];
        result_this = (uint64)kernel*(uint64)feature_345;
        result = (result_this << 8) + (result_prev >> 24);

        result_prev = result_this;

        output[col - 4] =  result & 255;
        output[col - 3] = (result >> 8) & 255;
        output[col - 2] = (result >> 16) & 255;
        output[col - 1] = (result >> 24) & 255;
    }
}

template<int F_DIM, int O_DIM>
void split_conv1d_32bit_p2q2(uint4* feature, uint32 kernel, uint10* output){

    uint32 feature_012 = 0;
    uint32 feature_345 = 0;
    uint64 result_prev = 0;
    uint64 result_this = 0;
    uint64 result = 0;

    feature_012 = ((uint32)feature[4] << 28) | ((uint32)feature[3] << 21) | ((uint32)feature[2] << 14) | ((uint32)feature[1] << 7) | (uint32)feature[0];
    result_prev = (uint64)kernel*(uint64)feature_012;

    for (int col = 5; col < O_DIM + 5; col += 5){
        feature_345 = ((uint32)feature[col + 4] << 28) | ((uint32)feature[col + 3] << 21) | ((uint32)feature[col + 2] << 14) | ((uint32)feature[col + 1] << 7) | (uint32)feature[col];
        result_this = (uint64)kernel*(uint64)feature_345;
        result = (result_this << 7) + (result_prev >> 28);

        result_prev = result_this;

        output[col - 5] =  result & 127;
        output[col - 4] = (result >> 7) & 127;
        output[col - 3] = (result >> 14) & 127;
        output[col - 2] = (result >> 21) & 127;
        output[col - 1] = (result >> 28) & 127;
    }
}

template<int F_DIM, int O_DIM>
void split_conv1d_32bit_p1q1(uint4* feature, uint32 kernel, uint10* output){

    uint32 feature_012 = 0;
    uint32 feature_345 = 0;
    uint64 result_prev = 0;
    uint64 result_this = 0;
    uint64 result = 0;

    feature_012 = ((uint32)feature[6] << 30) | ((uint32)feature[5] << 25) | ((uint32)feature[4] << 20) | ((uint32)feature[3] << 15) | ((uint32)feature[2] << 10) | ((uint32)feature[1] << 5) | (uint32)feature[0];
    result_prev = (uint64)kernel*(uint64)feature_012;

    for (int col = 7; col < O_DIM + 7; col += 7){
        feature_345 = ((uint32)feature[col + 6] << 30) | ((uint32)feature[col + 5] << 25) | ((uint32)feature[col + 4] << 20) | ((uint32)feature[col + 3] << 15) | ((uint32)feature[col + 2] << 10) | ((uint32)feature[col + 1] << 5) | (uint32)feature[col];
        result_this = (uint64)kernel*(uint64)feature_345;
        result = (result_this << 5) + (result_prev >> 30);

        result_prev = result_this;

        output[col - 7] =  result & 31;
        output[col - 6] = (result >> 5) & 31;
        output[col - 5] = (result >> 10) & 31;
        output[col - 4] = (result >> 15) & 31;
        output[col - 3] = (result >> 20) & 31;
        output[col - 2] = (result >> 25) & 31;
        output[col - 1] = (result >> 30) & 31;
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

template<int f_dim, int k_dim, int o_dim>
long long dsp_conv1d_test_signed(int a_bits, int w_bits, int p_bits, int q_bits, int reps, int* feature, int kernel[k_dim], int output_gen[o_dim]){

    int nkg[3] = {};
    get_nkg(a_bits, w_bits, p_bits, q_bits, nkg);
    int n_bits = nkg[0];
    int k_bits = nkg[1];
    int g_bits = nkg[2];
    int s_bits = p_bits + q_bits + g_bits;
    // cout << "p = " << p_bits << ", q = " << q_bits << ", n = " << n_bits << ", k = " << k_bits << ", g = " << g_bits << ", s = " << s_bits << endl;

    uint4 feature_4bit[f_dim] = {};
    uint4 kernel_4bit[k_dim] = {};
    uint10 output_4bit[o_dim] = {};

    for (int i = 0; i < f_dim; i ++){
        feature_4bit[i] = feature[i];
    }

    uint32 kernel_012 = 0;
    for (int i = 0; i < k_bits; i ++){
        kernel_012 |= (uint32)kernel[i] << s_bits*(k_bits-i-1);
    }

    long long ts = 0;
    long long tt = 0;
    if (p_bits == 8) {
        ts = nanos();
        for (int i = 0; i < reps; i++){
            split_conv1d_32bit_p8q8_signed<f_dim, o_dim>(feature_4bit, kernel_012, output_4bit);
        }
        tt = nanos();
    } else if (p_bits == 7) {
        ts = nanos();
        for (int i = 0; i < reps; i++){
            split_conv1d_32bit_p7q7_signed<f_dim, o_dim>(feature_4bit, kernel_012, output_4bit);
        }
        tt = nanos();
    } else if (p_bits == 6) {
        ts = nanos();
        for (int i = 0; i < reps; i++){
            split_conv1d_32bit_p6q6_signed<f_dim, o_dim>(feature_4bit, kernel_012, output_4bit);
        }
        tt = nanos();
    } else if (p_bits == 5) {
        ts = nanos();
        for (int i = 0; i < reps; i++){
            split_conv1d_32bit_p5q5_signed<f_dim, o_dim>(feature_4bit, kernel_012, output_4bit);
        }
        tt = nanos();
    } else if (p_bits == 4) {
        ts = nanos();
        for (int i = 0; i < reps; i++){
            split_conv1d_32bit_p4q4_signed<f_dim, o_dim>(feature_4bit, kernel_012, output_4bit);
        }
        tt = nanos();
    } else if (p_bits == 3) {
        ts = nanos();
        for (int i = 0; i < reps; i++){
            split_conv1d_32bit_p3q3_signed<f_dim, o_dim>(feature_4bit, kernel_012, output_4bit);
        }
        tt = nanos();
    } else if (p_bits == 2) {
        ts = nanos();
        for (int i = 0; i < reps; i++){
            split_conv1d_32bit_p2q2_signed<f_dim, o_dim>(feature_4bit, kernel_012, output_4bit);
        }
        tt = nanos();
    } else if (p_bits == 1) {
        ts = nanos();
        for (int i = 0; i < reps; i++){
            split_conv1d_32bit_p1q1_signed<f_dim, o_dim>(feature_4bit, kernel_012, output_4bit);
        }
        tt = nanos();
    } 

    #ifdef PRINT
    cout << "Output: " << endl;
    for (int i = 0; i < o_dim; i ++){
        int val = output_4bit[i];
        cout << val << " ";
    }
    cout << endl;
    #endif

    int count = 0;
    for (int i = 0; i < o_dim; i ++){
        int val = output_4bit[i];

        if (val >> (s_bits - 1) & 1 || val >> s_bits & 1) {
            val = val - pow(2, s_bits);
        } 
        if ((int)val != output_gen[i]) {
            count ++;
            // cout << "MISMATCH @ i = " << i << ", " << val << " vs. " << output_gen[i] << endl;
        }
    }
    
    if (count == 0) cout << "ALL TEST PASS" << endl;
    else cout << count << " TESTS FAIL" << endl;
    return (tt - ts)/reps;
}

template<int f_dim, int k_dim, int o_dim>
long long dsp_conv1d_test_unsigned(int a_bits, int w_bits, int p_bits, int q_bits, int reps, int* feature, int kernel[k_dim], int output_gen[o_dim]){

    int nkg[3] = {};
    get_nkg(a_bits, w_bits, p_bits, q_bits, nkg);
    int n_bits = nkg[0];
    int k_bits = nkg[1];
    int g_bits = nkg[2];
    int s_bits = p_bits + q_bits + g_bits;
    // cout << "p = " << p_bits << ", q = " << q_bits << ", n = " << n_bits << ", k = " << k_bits << ", g = " << g_bits << ", s = " << s_bits << endl;

    uint4 feature_4bit[f_dim] = {};
    uint4 kernel_4bit[k_dim] = {};
    uint10 output_4bit[o_dim] = {};

    for (int i = 0; i < f_dim; i ++){
        feature_4bit[i] = feature[i];
    }

    uint32 kernel_012 = 0;
    for (int i = 0; i < k_bits; i ++){
        kernel_012 |= (uint32)kernel[i] << s_bits*(k_bits-i-1);
    }

    long long ts = 0;
    long long tt = 0;
    if (p_bits == 8) {
        ts = nanos();
        for (int i = 0; i < reps; i++){
            split_conv1d_32bit_p8q8<f_dim, o_dim>(feature_4bit, kernel_012, output_4bit);
        }
        tt = nanos();
    } else if (p_bits == 7) {
        ts = nanos();
        for (int i = 0; i < reps; i++){
            split_conv1d_32bit_p7q7<f_dim, o_dim>(feature_4bit, kernel_012, output_4bit);
        }
        tt = nanos();
    } else if (p_bits == 6) {
        ts = nanos();
        for (int i = 0; i < reps; i++){
            split_conv1d_32bit_p6q6<f_dim, o_dim>(feature_4bit, kernel_012, output_4bit);
        }
        tt = nanos();
    } else if (p_bits == 5) {
        ts = nanos();
        for (int i = 0; i < reps; i++){
            split_conv1d_32bit_p5q5<f_dim, o_dim>(feature_4bit, kernel_012, output_4bit);
        }
        tt = nanos();
    } else if (p_bits == 4) {
        ts = nanos();
        for (int i = 0; i < reps; i++){
            split_conv1d_32bit_p4q4<f_dim, o_dim>(feature_4bit, kernel_012, output_4bit);
        }
        tt = nanos();
    } else if (p_bits == 3) {
        ts = nanos();
        for (int i = 0; i < reps; i++){
            split_conv1d_32bit_p3q3<f_dim, o_dim>(feature_4bit, kernel_012, output_4bit);
        }
        tt = nanos();
    } else if (p_bits == 2) {
        ts = nanos();
        for (int i = 0; i < reps; i++){
            split_conv1d_32bit_p2q2<f_dim, o_dim>(feature_4bit, kernel_012, output_4bit);
        }
        tt = nanos();
    } else if (p_bits == 1) {
        ts = nanos();
        for (int i = 0; i < reps; i++){
            split_conv1d_32bit_p1q1<f_dim, o_dim>(feature_4bit, kernel_012, output_4bit);
        }
        tt = nanos();
    } 

    #ifdef PRINT
    cout << "Output: " << endl;
    for (int i = 0; i < o_dim; i ++){
        int val = output_4bit[i];
        cout << val << " ";
    }
    cout << endl;
    #endif

    int count = 0;
    for (int i = 0; i < o_dim; i ++){
        int val = output_4bit[i];

        if (val >> (s_bits - 1) & 1 || val >> s_bits & 1) {
            val = val - pow(2, s_bits);
        } 
        if ((int)val != output_gen[i]) {
            count ++;
            // cout << "MISMATCH @ i = " << i << ", " << val << " vs. " << output_gen[i] << endl;
        }
    }
    
    if (count == 0) cout << "ALL TEST PASS" << endl;
    else cout << count << " TESTS FAIL" << endl;
    return (tt - ts)/reps;
}

int main(){

    srand(time(NULL));
    // srand(22);

    const int a_bits = 32;
    const int w_bits = 32;

    const int c_in = 1;
    const int c_out = 1;
    const int h_in = 3;
    const int w_in = 6;
    const int h_out = h_in - 2;
    const int w_out = w_in - 2;

    const int f_dim = 100000;
    const int reps = 1000;

    long long ts = 0;
    long long tt = 0;

    cout << "DSP_SPLIT" << endl << "======================================" << endl;

    int* feature = new int[f_dim];

    #ifdef PRINT
    cout << "Feature: ";
    #endif
    for (int i = 0; i < f_dim; i ++){
        #ifdef PRINT
        // feature[i] = i%3-3;
        feature[i] = rand()%5-5;
        cout << feature[i] << " ";
        #else
        feature[i] = rand()%2;
        #endif
    }

    cout << endl << "======================================" << endl;

    int p_bit, q_bit;
    long long elapses_gen, elapses_dsp;

    #ifdef TEST_1D
    cout << "Normal conv1d tests:" << endl;
    cout << "Running " << f_dim << "-size feature for " << reps << "times" << endl;
    
    const int k_dim_2 = 2;
    int kernel_2[k_dim_2] = {};
    #ifdef PRINT
    cout << endl << "Kernel: ";
    #endif
    for (int i = 0; i < k_dim_2; i ++){
        #ifdef PRINT
        kernel_2[i] = i + 1;
        cout << kernel_2[i] << " ";
        #else
        kernel_2[i] = rand() %2;
        #endif
    }
    cout << endl;
    const int o_dim_2 = f_dim - k_dim_2 + 1;
    int output_gen_2[o_dim_2] = {};    
    p_bit = 8;
    q_bit = 8;
    cout << "Testing " << p_bit << "bits:" << endl;
    elapses_gen = general_conv1d_test<f_dim, k_dim_2, o_dim_2>(reps, feature, kernel_2, output_gen_2);
    cout << "Gen Elapsed: " << elapses_gen << " ns" << endl;
    cout << "======================================" << endl;
    elapses_dsp = dsp_conv1d_test_signed<f_dim, k_dim_2, o_dim_2>(a_bits, w_bits, p_bit, q_bit, reps, feature, kernel_2, output_gen_2);
    cout << "Signed Elapsed: " << elapses_dsp << " ns" << endl;
    cout << "======================================" << endl;
    elapses_dsp = dsp_conv1d_test_unsigned<f_dim, k_dim_2, o_dim_2>(a_bits, w_bits, p_bit, q_bit, reps, feature, kernel_2, output_gen_2);
    cout << "Unsigned Elapsed: " << elapses_dsp << " ns" << endl;
    cout << "======================================" << endl << endl;

    p_bit = p_bit - 1;
    q_bit = q_bit - 1;
    cout << "Testing " << p_bit << "bits:" << endl;
    elapses_gen = general_conv1d_test<f_dim, k_dim_2, o_dim_2>(reps, feature, kernel_2, output_gen_2);
    cout << "Gen Elapsed: " << elapses_gen << " ns" << endl;
    cout << "======================================" << endl;
    elapses_dsp = dsp_conv1d_test_signed<f_dim, k_dim_2, o_dim_2>(a_bits, w_bits, p_bit, q_bit, reps, feature, kernel_2, output_gen_2);
    cout << "Signed Elapsed: " << elapses_dsp << " ns" << endl;
    cout << "======================================" << endl;
    elapses_dsp = dsp_conv1d_test_unsigned<f_dim, k_dim_2, o_dim_2>(a_bits, w_bits, p_bit, q_bit, reps, feature, kernel_2, output_gen_2);
    cout << "Unsigned Elapsed: " << elapses_dsp << " ns" << endl;
    cout << "======================================" << endl << endl;

    const int k_dim_3 = 3;
    int kernel_3[k_dim_3] = {};
    #ifdef PRINT
    cout << endl << "Kernel: ";
    #endif
    for (int i = 0; i < k_dim_3; i ++){
        #ifdef PRINT
        kernel_3[i] = i + 1;
        cout << kernel_3[i] << " ";
        #else
        kernel_3[i] = rand() %2;
        #endif
    }
    cout << endl;
    const int o_dim_3 = f_dim - k_dim_3 + 1;
    int output_gen_3[o_dim_3] = {};

    p_bit = p_bit - 1;
    q_bit = q_bit - 1;
    cout << "Testing " << p_bit << "bits:" << endl;
    elapses_gen = general_conv1d_test<f_dim, k_dim_3, o_dim_3>(reps, feature, kernel_3, output_gen_3);
    cout << "Gen Elapsed: " << elapses_gen << " ns" << endl;
    cout << "======================================" << endl;
    elapses_dsp = dsp_conv1d_test_signed<f_dim, k_dim_3, o_dim_3>(a_bits, w_bits, p_bit, q_bit, reps, feature, kernel_3, output_gen_3);
    cout << "Signed Elapsed: " << elapses_dsp << " ns" << endl;
    cout << "======================================" << endl;
    elapses_dsp = dsp_conv1d_test_unsigned<f_dim, k_dim_3, o_dim_3>(a_bits, w_bits, p_bit, q_bit, reps, feature, kernel_3, output_gen_3);
    cout << "Unsigned Elapsed: " << elapses_dsp << " ns" << endl;
    cout << "======================================" << endl << endl;
    
    p_bit = p_bit - 1;
    q_bit = q_bit - 1;
    cout << "Testing " << p_bit << "bits:" << endl;
    elapses_gen = general_conv1d_test<f_dim, k_dim_3, o_dim_3>(reps, feature, kernel_3, output_gen_3);
    cout << "Gen Elapsed: " << elapses_gen << " ns" << endl;
    cout << "======================================" << endl;
    elapses_dsp = dsp_conv1d_test_signed<f_dim, k_dim_3, o_dim_3>(a_bits, w_bits, p_bit, q_bit, reps, feature, kernel_3, output_gen_3);
    cout << "Signed Elapsed: " << elapses_dsp << " ns" << endl;
    cout << "======================================" << endl;
    elapses_dsp = dsp_conv1d_test_unsigned<f_dim, k_dim_3, o_dim_3>(a_bits, w_bits, p_bit, q_bit, reps, feature, kernel_3, output_gen_3);
    cout << "Unsigned Elapsed: " << elapses_dsp << " ns" << endl;
    cout << "======================================" << endl << endl;

    p_bit = p_bit - 1;
    q_bit = q_bit - 1;
    cout << "Testing " << p_bit << "bits:" << endl;
    elapses_gen = general_conv1d_test<f_dim, k_dim_3, o_dim_3>(reps, feature, kernel_3, output_gen_3);
    cout << "Gen Elapsed: " << elapses_gen << " ns" << endl;
    cout << "======================================" << endl;
    elapses_dsp = dsp_conv1d_test_signed<f_dim, k_dim_3, o_dim_3>(a_bits, w_bits, p_bit, q_bit, reps, feature, kernel_3, output_gen_3);
    cout << "Signed Elapsed: " << elapses_dsp << " ns" << endl;
    cout << "======================================" << endl;
    elapses_dsp = dsp_conv1d_test_unsigned<f_dim, k_dim_3, o_dim_3>(a_bits, w_bits, p_bit, q_bit, reps, feature, kernel_3, output_gen_3);
    cout << "Unsigned Elapsed: " << elapses_dsp << " ns" << endl;
    cout << "======================================" << endl << endl;

    const int k_dim_4 = 4;
    int kernel_4[k_dim_4] = {};
    #ifdef PRINT
    cout << endl << "Kernel: ";
    #endif
    for (int i = 0; i < k_dim_4; i ++){
        #ifdef PRINT
        kernel_4[i] = i + 1;
        cout << kernel_4[i] << " ";
        #else
        kernel_4[i] = rand() %2;
        #endif
    }
    cout << endl;
    const int o_dim_4 = f_dim - k_dim_4 + 1;
    int output_gen_4[o_dim_4] = {};
    p_bit = p_bit - 1;
    q_bit = q_bit - 1;
    cout << "Testing " << p_bit << "bits:" << endl;
    elapses_gen = general_conv1d_test<f_dim, k_dim_4, o_dim_4>(reps, feature, kernel_4, output_gen_4);
    cout << "Gen Elapsed: " << elapses_gen << " ns" << endl;
    cout << "======================================" << endl;
    elapses_dsp = dsp_conv1d_test_signed<f_dim, k_dim_4, o_dim_4>(a_bits, w_bits, p_bit, q_bit, reps, feature, kernel_4, output_gen_4);
    cout << "Signed Elapsed: " << elapses_dsp << " ns" << endl;
    cout << "======================================" << endl;
    elapses_dsp = dsp_conv1d_test_unsigned<f_dim, k_dim_4, o_dim_4>(a_bits, w_bits, p_bit, q_bit, reps, feature, kernel_4, output_gen_4);
    cout << "Unsigned Elapsed: " << elapses_dsp << " ns" << endl;
    cout << "======================================" << endl << endl;

    const int k_dim_5 = 5;
    int kernel_5[k_dim_5] = {};
    #ifdef PRINT
    cout << endl << "Kernel: ";
    #endif
    for (int i = 0; i < k_dim_5; i ++){
        #ifdef PRINT
        kernel_5[i] = i + 1;
        cout << kernel_5[i] << " ";
        #else
        kernel_5[i] = rand() %2;
        #endif
    }
    cout << endl;
    const int o_dim_5 = f_dim - k_dim_5 + 1;
    int output_gen_5[o_dim_5] = {};
    p_bit = p_bit - 1;
    q_bit = q_bit - 1;
    cout << "Testing " << p_bit << "bits:" << endl;
    elapses_gen = general_conv1d_test<f_dim, k_dim_5, o_dim_5>(reps, feature, kernel_5, output_gen_5);
    cout << "Gen Elapsed: " << elapses_gen << " ns" << endl;
    cout << "======================================" << endl;   
    elapses_dsp = dsp_conv1d_test_signed<f_dim, k_dim_5, o_dim_5>(a_bits, w_bits, p_bit, q_bit, reps, feature, kernel_5, output_gen_5);
    cout << "Signed Elapsed: " << elapses_dsp << " ns" << endl;
    cout << "======================================" << endl;
    elapses_dsp = dsp_conv1d_test_unsigned<f_dim, k_dim_5, o_dim_5>(a_bits, w_bits, p_bit, q_bit, reps, feature, kernel_5, output_gen_5);
    cout << "Unsigned Elapsed: " << elapses_dsp << " ns" << endl;
    cout << "======================================" << endl << endl;
    
    const int k_dim_7 = 7;
    int kernel_7[k_dim_7] = {};
    #ifdef PRINT
    cout << endl << "Kernel: ";
    #endif
    for (int i = 0; i < k_dim_7; i ++){
        #ifdef PRINT
        kernel_7[i] = i + 1;
        cout << kernel_7[i] << " ";
        #else
        kernel_7[i] = rand() %2;
        #endif
    }
    cout << endl;
    const int o_dim_7 = f_dim - k_dim_7 + 1;
    int output_gen_7[o_dim_7] = {};
    p_bit = p_bit - 1;
    q_bit = q_bit - 1;
    cout << "Testing " << p_bit << "bits:" << endl;
    elapses_gen = general_conv1d_test<f_dim, k_dim_7, o_dim_7>(reps, feature, kernel_7, output_gen_7);
    cout << "Gen Elapsed: " << elapses_gen << " ns" << endl;
    cout << "======================================" << endl;
    elapses_dsp = dsp_conv1d_test_signed<f_dim, k_dim_7, o_dim_7>(a_bits, w_bits, p_bit, q_bit, reps, feature, kernel_7, output_gen_7);
    cout << "Signed Elapsed: " << elapses_dsp << " ns" << endl;
    cout << "======================================" << endl;
    elapses_dsp = dsp_conv1d_test_unsigned<f_dim, k_dim_7, o_dim_7>(a_bits, w_bits, p_bit, q_bit, reps, feature, kernel_7, output_gen_7);
    cout << "Unsigned Elapsed: " << elapses_dsp << " ns" << endl;
    cout << "======================================" << endl << endl;

    #endif
    return 0;
}