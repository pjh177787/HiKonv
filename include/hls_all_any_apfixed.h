// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
/*****************************************************************************
 *
 *     Author: Xilinx, Inc.
 *
 *     This text contains proprietary, confidential information of
 *     Xilinx, Inc. , is distributed by under license from Xilinx,
 *     Inc., and may be used, copied and/or disclosed only pursuant to
 *     the terms of a valid license agreement with Xilinx, Inc.
 *
 *     XILINX IS PROVIDING THIS DESIGN, CODE, OR INFORMATION "AS IS"
 *     AS A COURTESY TO YOU, SOLELY FOR USE IN DEVELOPING PROGRAMS AND
 *     SOLUTIONS FOR XILINX DEVICES.  BY PROVIDING THIS DESIGN, CODE,
 *     OR INFORMATION AS ONE POSSIBLE IMPLEMENTATION OF THIS FEATURE,
 *     APPLICATION OR STANDARD, XILINX IS MAKING NO REPRESENTATION
 *     THAT THIS IMPLEMENTATION IS FREE FROM ANY CLAIMS OF INFRINGEMENT,
 *     AND YOU ARE RESPONSIBLE FOR OBTAINING ANY RIGHTS YOU MAY REQUIRE
 *     FOR YOUR IMPLEMENTATION.  XILINX EXPRESSLY DISCLAIMS ANY
 *     WARRANTY WHATSOEVER WITH RESPECT TO THE ADEQUACY OF THE
 *     IMPLEMENTATION, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OR
 *     REPRESENTATIONS THAT THIS IMPLEMENTATION IS FREE FROM CLAIMS OF
 *     INFRINGEMENT, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *     FOR A PARTICULAR PURPOSE.
 *
 *     Xilinx products are not intended for use in life support appliances,
 *     devices, or systems. Use in such applications is expressly prohibited.
 *
 *     (c) Copyright 2008-2012 Xilinx Inc.
 *     All rights reserved.
 *
 *****************************************************************************/
#ifndef _X_HLS_ALL_ANY_APFIXED_H_
#define _X_HLS_ALL_ANY_APFIXED_H_
#include "ap_fixed.h"
#include "ap_int.h"

///hls_all ap_fixed
template <int W_, int I_>
bool generic_all(ap_fixed<W_,I_> x)
{
#pragma HLS pipeline II=1
    for (int i=0; i<W_;i++)
    	if (x[i]==0)
    		return false;
    return true;
}

///hls_all ap_ufixed
template <int W_, int I_>
bool generic_all(ap_ufixed<W_,I_> x)
{
#pragma HLS pipeline II=1
    for (int i=0; i<W_;i++)
    	if (x[i]==0)
    		return false;
    return true;
}

///hls_all ap_int
template <int I_>
bool generic_all(ap_int<I_> x)
{
#pragma HLS pipeline II=1
    for (int i=0; i<I_;i++)
    	if (x[i]==0)
    		return false;
    return true;
}

///hls_all ap_uint
template <int I_>
bool generic_all(ap_uint<I_> x)
{
#pragma HLS pipeline II=1
    for (int i=0; i<I_;i++)
    	if (x[i]==0)
    		return false;
    return true;
}

///hls_any ap_fixed
template <int W_, int I_>
bool generic_any(ap_fixed<W_,I_> x)
{
#pragma HLS pipeline II=1
    for (int i=0; i<W_;i++)
    	if (x[i]==1)
    		return true;
    return false;
}

///hls_any ap_ufixed
template <int W_, int I_>
bool generic_any(ap_ufixed<W_,I_> x)
{
#pragma HLS pipeline II=1
    for (int i=0; i<W_;i++)
    	if (x[i]==1)
    		return true;
    return false;
}

///hls_any ap_int
template <int I_>
bool generic_any(ap_int<I_> x)
{
#pragma HLS pipeline II=1
    for (int i=0; i<I_;i++)
    	if (x[i]==1)
    		return true;
    return false;
}

///hls_any ap_uint
template <int I_>
bool generic_any(ap_uint<I_> x)
{
#pragma HLS pipeline II=1
    for (int i=0; i<I_;i++)
    	if (x[i]==1)
    		return true;
    return false;
}

#endif



