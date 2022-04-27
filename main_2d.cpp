#pragma GCC optimize(0)

#include <iostream>
#include <ap_fixed.h>
#include <ap_int.h>
#include <cmath>
#include <chrono>
#include <stdlib.h>
using namespace std;

#define TEST_2D
#define TEST_SIGNED

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
void split_conv2d_32bit_unsigned(uint4 feature[CH_IN][H_IN][W_IN], uint32 kernel[CH_OUT][CH_IN][3], uint10 output[CH_OUT][H_OUT][W_OUT]){

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

                        #ifdef PRINT_PROC
                        cout << "c_out: " << c_out << " c_in: " << c_in << " row: " << row << " k_row: " << k_row << " col: " << col << endl;
                        cout << "feature_012 :";
                        print_bin(feature_012, 10);
                        cout << "feature_345 :";
                        print_bin(feature_345, 10);
                        cout << "Kernel_012 :";
                        print_bin(kernel_temp, 10);
                        cout << "Result_prev   :";
                        print_bin(result_prev, 10);
                        cout << "Result_this   :";
                        print_bin(result_this, 10);
                        cout << "Result     :";
                        print_bin(result, 10);
                        #endif

                        result_prev = result_this;
                        
                        output_temp[col - 3] =  result & 1023;
                        output_temp[col - 2] = (result >> 10) & 1023;
                        output_temp[col - 1] = (result >> 20) & 1023;
                        #ifdef PRINT_PROC
                        cout << "Output 0:" << output_temp[col - 3] << endl;
                        cout << "Output 1:" << output_temp[col - 2] << endl;
                        cout << "Output 2:" << output_temp[col - 1] << endl;
                        cout << endl;
                        #endif
                    }
                    for (int col = 0; col < W_OUT; col ++){
                        output[c_out][row][col] += output_temp[col];
                    }
                }
            }
        }
    }
}

template<int CH_IN, int CH_OUT, int H_IN, int W_IN, int H_OUT, int W_OUT>
void split_conv2d_32bit_signed(uint4 feature[CH_IN][H_IN][W_IN], uint32 kernel[CH_OUT][CH_IN][3], uint10 output[CH_OUT][H_OUT][W_OUT]){

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
                        
                        int sign_3 = (result_prev >> 19) & 1;
                        int sign_2 = (result >> 9) & 1;
                        int sign_1 = (result >> 19) & 1;
                        output_temp[col - 3] =  (result & 1023) + sign_3;
                        output_temp[col - 2] = ((result >> 10) & 1023) + sign_2;
                        output_temp[col - 1] = ((result >> 20) & 1023) + sign_1;
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

int main(){

    const int a_bits = 32;
    const int w_bits = 32;

    const int c_in = 64;
    const int c_out = 36;
    const int h_in = 10;
    const int w_in = 20;
    const int h_out = h_in - 2;
    const int w_out = w_in - 2;

    const int f_dim = 100000;


    const int reps = 1;
    int count = 0;

    long long ts = 0;
    long long tt = 0;

    cout << "DSP_SPLIT" << endl << "======================================" << endl;

    int* feature = new int[f_dim];

    #ifdef PRINT
    cout << "Feature: ";
    #endif
    for (int i = 0; i < f_dim; i ++){
        #ifdef PRINT
        feature[i] = i + 1;
        cout << feature[i] << " ";
        #else
        feature[i] = rand() %2;
        #endif
    }

    int feature3d[c_in][h_in][w_in] = {};
    int kernel3d[c_out][c_in][3][3] = {};
    int output3d[c_out][h_out][w_out] = {};

    #ifdef PRINT
    cout << endl<< "Feature2d: " << endl;
    #endif
    for (int i = 0; i < c_in; i ++){
        for (int j = 0; j < h_in; j ++){
            for (int k = 0; k < w_in; k ++){
                #ifdef PRINT
                feature3d[i][j][k] = (i*h_in*w_in + j*w_in + k + 1)%8;
                cout << feature3d[i][j][k] << " ";
                #else
                feature3d[i][j][k] = rand() %7;
                #endif
            }
            #ifdef PRINT
            cout << endl;
            #endif
        }
        #ifdef PRINT
        cout << endl;
        #endif
    }

    #ifdef PRINT
    cout << endl << " Kernel2d: " << endl;
    #endif
    for (int i = 0; i < c_out; i ++){
        for (int j = 0; j < c_in; j ++){
            for (int k = 0; k < 3; k ++){
                for (int l = 0; l < 3; l ++){
                    #ifdef PRINT
                    kernel3d[i][j][k][l] = (i*c_in*3*3 + j*3*3 + k*3 + l + 1)%8;
                    cout << kernel3d[i][j][k][l] << ", ";
                    #else
                    kernel3d[i][j][k][l] = rand() %2;
                    #endif
                }
                #ifdef PRINT
                cout << endl;
                #endif
            }
            #ifdef PRINT
            cout << endl;
            #endif
        }
        #ifdef PRINT
        cout << endl;
        #endif
    }

    uint4 feature3d_4bit[c_in][h_in][w_in] = {};
    uint4 kernel3d_4bit[c_out][c_in][3][3] = {};

    for (int i = 0; i < c_in; i ++){
        for (int j = 0; j < h_in; j ++){
            for (int k = 0; k < w_in; k ++){
                feature3d_4bit[i][j][k] = feature3d[i][j][k];
            }
        }
    }
    for (int i = 0; i < c_out; i ++){
        for (int j = 0; j < c_in; j ++){
            for (int k = 0; k < 3; k ++){
                for (int l = 0; l < 3; l ++){
                    kernel3d_4bit[i][j][k][l] = kernel3d[i][j][k][l];
                }
            }
        }
    }

    uint32 kernel3d_012[c_out][c_in][3] = {};
    for (int i = 0; i < c_out; i ++){
        for (int j = 0; j < c_in; j ++){
            for (int k = 0; k < 3; k ++){
                uint32 kernel3d_012_temp = 0;
                kernel3d_012_temp |= (uint32)kernel3d_4bit[i][j][k][0] << 20;
                kernel3d_012_temp |= (uint32)kernel3d_4bit[i][j][k][1] << 10;
                kernel3d_012_temp |= (uint32)kernel3d_4bit[i][j][k][2];
                kernel3d_012[i][j][k] = kernel3d_012_temp;
            }
        }
    }

    // estimate_speedup<o_dim>(p_bits, q_bits, 32, 32, reps, output);
    cout << "======================================" << endl;
    
    #ifdef TEST_2D
    cout << "Normal conv2d tests:" << endl;
    ts = nanos();
    for (int i = 0; i < reps; i ++){
        general_conv2d<c_in, c_out, h_in, w_in, h_out, w_out>(feature3d, kernel3d, output3d);
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

    uint10 output3d_4bit[c_out][h_out][w_out] = {};

    #ifdef TEST_SIGNED
    cout << "dsp conv2d test signed:" << endl;
    ts = nanos();
    for (int i = 0; i < reps; i++){
        split_conv2d_32bit_signed<c_in, c_out, h_in, w_in, h_out, w_out>(feature3d_4bit, kernel3d_012, output3d_4bit);
    }
    tt = nanos();

    #ifdef PRINT
    cout << "Output: " << endl;
    for (int i = 0; i < c_out; i ++){
        for (int j = 0; j < h_out; j ++){
            for (int k = 0; k < w_out; k ++){
                cout << output3d_4bit[i][j][k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    #endif

    cout << "Elapsed: " << (tt - ts)/reps << " ns" << endl;
    count = 0;
    for (int i = 0; i < c_out; i ++){
        for (int j = 0; j < h_out; j ++){
            for (int k = 0; k < w_out; k ++){
                if ((int)output3d_4bit[i][j][k] != output3d[i][j][k]) {
                    count ++;
                    cout << "MISMATCH @ (i, j, k) = (" << i << ", " << j << ", " << k << ")  " << output3d_4bit[i][j][k] << " vs. " << output3d[i][j][k] << endl;
                }
            }
        }
    }
    
    if (count == 0) cout << "ALL TEST PASS" << endl;
    else cout << count << " TESTS FAIL" << endl;
    cout << "======================================" << endl << endl;
    #else
    cout << "dsp conv2d test unsigned:" << endl;
    ts = nanos();
    for (int i = 0; i < reps; i++){
        split_conv2d_32bit_unsigned<c_in, c_out, h_in, w_in, h_out, w_out>(feature3d_4bit, kernel3d_012, output3d_4bit);
    }
    tt = nanos();

    #ifdef PRINT
    cout << "Output: " << endl;
    for (int i = 0; i < c_out; i ++){
        for (int j = 0; j < h_out; j ++){
            for (int k = 0; k < w_out; k ++){
                cout << output3d_4bit[i][j][k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    #endif

    cout << "Elapsed: " << (tt - ts)/reps << " ns" << endl;
    count = 0;
    for (int i = 0; i < c_out; i ++){
        for (int j = 0; j < h_out; j ++){
            for (int k = 0; k < w_out; k ++){
                if ((int)output3d_4bit[i][j][k] != output3d[i][j][k]) {
                    count ++;
                    cout << "MISMATCH @ (i, j, k) = (" << i << ", " << j << ", " << k << ")  " << output3d_4bit[i][j][k] << " vs. " << output3d[i][j][k] << endl;
                }
            }
        }
    }
    
    if (count == 0) cout << "ALL TEST PASS" << endl;
    else cout << count << " TESTS FAIL" << endl;
    cout << "======================================" << endl << endl;
    #endif
    return 0;
}