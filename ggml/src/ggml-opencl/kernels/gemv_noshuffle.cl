#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#endif

// assume
#define QK4_0 32
#ifndef N_SIMDGROUP
#define N_SIMDGROUP 16
#endif

#define dequantizeBlockAccum_ns_sgbroadcast_1_hi(total_sums, bits4, scale, y) \
    float shared_y; \
    float2 block_sum = (float2)(0.0f); \
    shared_y = sub_group_broadcast(y.s0, 0); \
    block_sum.s0 += ((bits4.s0 & 0x000F) - 8) * shared_y; \
    block_sum.s1 += ((bits4.s1 & 0x000F) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 0); \
    block_sum.s0 += (((bits4.s0 & 0x00F0) >> 4) - 8) * shared_y; \
    block_sum.s1 += (((bits4.s1 & 0x00F0) >> 4) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 0); \
    block_sum.s0 += (((bits4.s0 & 0x0F00) >> 8) - 8) * shared_y; \
    block_sum.s1 += (((bits4.s1 & 0x0F00) >> 8) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 0); \
    block_sum.s0 += (((bits4.s0 & 0xF000) >> 12) - 8) * shared_y; \
    block_sum.s1 += (((bits4.s1 & 0xF000) >> 12) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 0); \
    block_sum.s0 += ((bits4.s2 & 0x000F) - 8) * shared_y; \
    block_sum.s1 += ((bits4.s3 & 0x000F) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 0); \
    block_sum.s0 += (((bits4.s2 & 0x00F0) >> 4) - 8) * shared_y; \
    block_sum.s1 += (((bits4.s3 & 0x00F0) >> 4) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 0); \
    block_sum.s0 += (((bits4.s2 & 0x0F00) >> 8) - 8) * shared_y; \
    block_sum.s1 += (((bits4.s3 & 0x0F00) >> 8) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 0); \
    block_sum.s0 += (((bits4.s2 & 0xF000) >> 12) - 8) * shared_y; \
    block_sum.s1 += (((bits4.s3 & 0xF000) >> 12) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 1); \
    block_sum.s0 += ((bits4.s4 & 0x000F) - 8) * shared_y; \
    block_sum.s1 += ((bits4.s5 & 0x000F) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 1); \
    block_sum.s0 += (((bits4.s4 & 0x00F0) >> 4) - 8) * shared_y; \
    block_sum.s1 += (((bits4.s5 & 0x00F0) >> 4) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 1); \
    block_sum.s0 += (((bits4.s4 & 0x0F00) >> 8) - 8) * shared_y; \
    block_sum.s1 += (((bits4.s5 & 0x0F00) >> 8) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 1); \
    block_sum.s0 += (((bits4.s4 & 0xF000) >> 12) - 8) * shared_y; \
    block_sum.s1 += (((bits4.s5 & 0xF000) >> 12) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 1); \
    block_sum.s0 += ((bits4.s6 & 0x000F) - 8) * shared_y; \
    block_sum.s1 += ((bits4.s7 & 0x000F) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 1); \
    block_sum.s0 += (((bits4.s6 & 0x00F0) >> 4) - 8) * shared_y; \
    block_sum.s1 += (((bits4.s7 & 0x00F0) >> 4) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 1); \
    block_sum.s0 += (((bits4.s6 & 0x0F00) >> 8) - 8) * shared_y; \
    block_sum.s1 += (((bits4.s7 & 0x0F00) >> 8) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 1); \
    block_sum.s0 += (((bits4.s6 & 0xF000) >> 12) - 8) * shared_y; \
    block_sum.s1 += (((bits4.s7 & 0xF000) >> 12) - 8) * shared_y; \


#define dequantizeBlockAccum_ns_sgbroadcast_1_lo(total_sums, bits4, scale, y) \
    shared_y = sub_group_broadcast(y.s0, 2); \
    block_sum.s0 += ((bits4.s0 & 0x000F) - 8) * shared_y; \
    block_sum.s1 += ((bits4.s1 & 0x000F) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 2); \
    block_sum.s0 += (((bits4.s0 & 0x00F0) >> 4) - 8) * shared_y; \
    block_sum.s1 += (((bits4.s1 & 0x00F0) >> 4) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 2); \
    block_sum.s0 += (((bits4.s0 & 0x0F00) >> 8) - 8) * shared_y; \
    block_sum.s1 += (((bits4.s1 & 0x0F00) >> 8) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 2); \
    block_sum.s0 += (((bits4.s0 & 0xF000) >> 12) - 8) * shared_y; \
    block_sum.s1 += (((bits4.s1 & 0xF000) >> 12) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 2); \
    block_sum.s0 += ((bits4.s2 & 0x000F) - 8) * shared_y; \
    block_sum.s1 += ((bits4.s3 & 0x000F) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 2); \
    block_sum.s0 += (((bits4.s2 & 0x00F0) >> 4) - 8) * shared_y; \
    block_sum.s1 += (((bits4.s3 & 0x00F0) >> 4) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 2); \
    block_sum.s0 += (((bits4.s2 & 0x0F00) >> 8) - 8) * shared_y; \
    block_sum.s1 += (((bits4.s3 & 0x0F00) >> 8) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 2); \
    block_sum.s0 += (((bits4.s2 & 0xF000) >> 12) - 8) * shared_y; \
    block_sum.s1 += (((bits4.s3 & 0xF000) >> 12) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 3); \
    block_sum.s0 += ((bits4.s4 & 0x000F) - 8) * shared_y; \
    block_sum.s1 += ((bits4.s5 & 0x000F) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 3); \
    block_sum.s0 += (((bits4.s4 & 0x00F0) >> 4) - 8) * shared_y; \
    block_sum.s1 += (((bits4.s5 & 0x00F0) >> 4) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 3); \
    block_sum.s0 += (((bits4.s4 & 0x0F00) >> 8) - 8) * shared_y; \
    block_sum.s1 += (((bits4.s5 & 0x0F00) >> 8) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 3); \
    block_sum.s0 += (((bits4.s4 & 0xF000) >> 12) - 8) * shared_y; \
    block_sum.s1 += (((bits4.s5 & 0xF000) >> 12) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 3); \
    block_sum.s0 += ((bits4.s6 & 0x000F) - 8) * shared_y; \
    block_sum.s1 += ((bits4.s7 & 0x000F) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 3); \
    block_sum.s0 += (((bits4.s6 & 0x00F0) >> 4) - 8) * shared_y; \
    block_sum.s1 += (((bits4.s7 & 0x00F0) >> 4) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 3); \
    block_sum.s0 += (((bits4.s6 & 0x0F00) >> 8) - 8) * shared_y; \
    block_sum.s1 += (((bits4.s7 & 0x0F00) >> 8) - 8) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 3); \
    block_sum.s0 += (((bits4.s6 & 0xF000) >> 12) - 8) * shared_y; \
    block_sum.s1 += (((bits4.s7 & 0xF000) >> 12) - 8) * shared_y; \
    total_sums += block_sum * convert_float2(scale); \


#define dequantizeBlockAccum_ns_sgbroadcast_8_hi(total_sums, bits4, scale, y) \
    float8 shared_y; \
    float2 block_sum = (float2)(0.0f); \
    shared_y = sub_group_broadcast(y, 0); \
    block_sum.s0 += ((bits4.s0 & 0x000F) - 8) * shared_y.s0; \
    block_sum.s0 += (((bits4.s0 & 0x00F0) >> 4) - 8) * shared_y.s1; \
    block_sum.s0 += (((bits4.s0 & 0x0F00) >> 8) - 8) * shared_y.s2; \
    block_sum.s0 += (((bits4.s0 & 0xF000) >> 12) - 8) * shared_y.s3; \
    block_sum.s0 += ((bits4.s2 & 0x000F) - 8) * shared_y.s4; \
    block_sum.s0 += (((bits4.s2 & 0x00F0) >> 4) - 8) * shared_y.s5; \
    block_sum.s0 += (((bits4.s2 & 0x0F00) >> 8) - 8) * shared_y.s6; \
    block_sum.s0 += (((bits4.s2 & 0xF000) >> 12) - 8) * shared_y.s7; \
    block_sum.s1 += ((bits4.s1 & 0x000F) - 8) * shared_y.s0; \
    block_sum.s1 += (((bits4.s1 & 0x00F0) >> 4) - 8) * shared_y.s1; \
    block_sum.s1 += (((bits4.s1 & 0x0F00) >> 8) - 8) * shared_y.s2; \
    block_sum.s1 += (((bits4.s1 & 0xF000) >> 12) - 8) * shared_y.s3; \
    block_sum.s1 += ((bits4.s3 & 0x000F) - 8) * shared_y.s4; \
    block_sum.s1 += (((bits4.s3 & 0x00F0) >> 4) - 8) * shared_y.s5; \
    block_sum.s1 += (((bits4.s3 & 0x0F00) >> 8) - 8) * shared_y.s6; \
    block_sum.s1 += (((bits4.s3 & 0xF000) >> 12) - 8) * shared_y.s7; \
    shared_y = sub_group_broadcast(y, 1); \
    block_sum.s0 += ((bits4.s4 & 0x000F) - 8) * shared_y.s0; \
    block_sum.s0 += (((bits4.s4 & 0x00F0) >> 4) - 8) * shared_y.s1; \
    block_sum.s0 += (((bits4.s4 & 0x0F00) >> 8) - 8) * shared_y.s2; \
    block_sum.s0 += (((bits4.s4 & 0xF000) >> 12) - 8) * shared_y.s3; \
    block_sum.s0 += ((bits4.s6 & 0x000F) - 8) * shared_y.s4; \
    block_sum.s0 += (((bits4.s6 & 0x00F0) >> 4) - 8) * shared_y.s5; \
    block_sum.s0 += (((bits4.s6 & 0x0F00) >> 8) - 8) * shared_y.s6; \
    block_sum.s0 += (((bits4.s6 & 0xF000) >> 12) - 8) * shared_y.s7; \
    block_sum.s1 += ((bits4.s5 & 0x000F) - 8) * shared_y.s0; \
    block_sum.s1 += (((bits4.s5 & 0x00F0) >> 4) - 8) * shared_y.s1; \
    block_sum.s1 += (((bits4.s5 & 0x0F00) >> 8) - 8) * shared_y.s2; \
    block_sum.s1 += (((bits4.s5 & 0xF000) >> 12) - 8) * shared_y.s3; \
    block_sum.s1 += ((bits4.s7 & 0x000F) - 8) * shared_y.s4; \
    block_sum.s1 += (((bits4.s7 & 0x00F0) >> 4) - 8) * shared_y.s5; \
    block_sum.s1 += (((bits4.s7 & 0x0F00) >> 8) - 8) * shared_y.s6; \
    block_sum.s1 += (((bits4.s7 & 0xF000) >> 12) - 8) * shared_y.s7; \
    total_sums += block_sum * convert_float2(scale); \


#define dequantizeBlockAccum_ns_sgbroadcast_8_lo(total_sums, bits4, scale, y) \
    shared_y = sub_group_broadcast(y, 2); \
    block_sum.s0 += ((bits4.s0 & 0x000F) - 8) * shared_y.s0; \
    block_sum.s0 += (((bits4.s0 & 0x00F0) >> 4) - 8) * shared_y.s1; \
    block_sum.s0 += (((bits4.s0 & 0x0F00) >> 8) - 8) * shared_y.s2; \
    block_sum.s0 += (((bits4.s0 & 0xF000) >> 12) - 8) * shared_y.s3; \
    block_sum.s0 += ((bits4.s2 & 0x000F) - 8) * shared_y.s4; \
    block_sum.s0 += (((bits4.s2 & 0x00F0) >> 4) - 8) * shared_y.s5; \
    block_sum.s0 += (((bits4.s2 & 0x0F00) >> 8) - 8) * shared_y.s6; \
    block_sum.s0 += (((bits4.s2 & 0xF000) >> 12) - 8) * shared_y.s7; \
    block_sum.s1 += ((bits4.s1 & 0x000F) - 8) * shared_y.s0; \
    block_sum.s1 += (((bits4.s1 & 0x00F0) >> 4) - 8) * shared_y.s1; \
    block_sum.s1 += (((bits4.s1 & 0x0F00) >> 8) - 8) * shared_y.s2; \
    block_sum.s1 += (((bits4.s1 & 0xF000) >> 12) - 8) * shared_y.s3; \
    block_sum.s1 += ((bits4.s3 & 0x000F) - 8) * shared_y.s4; \
    block_sum.s1 += (((bits4.s3 & 0x00F0) >> 4) - 8) * shared_y.s5; \
    block_sum.s1 += (((bits4.s3 & 0x0F00) >> 8) - 8) * shared_y.s6; \
    block_sum.s1 += (((bits4.s3 & 0xF000) >> 12) - 8) * shared_y.s7; \
    shared_y = sub_group_broadcast(y, 3); \
    block_sum.s0 += ((bits4.s4 & 0x000F) - 8) * shared_y.s0; \
    block_sum.s0 += (((bits4.s4 & 0x00F0) >> 4) - 8) * shared_y.s1; \
    block_sum.s0 += (((bits4.s4 & 0x0F00) >> 8) - 8) * shared_y.s2; \
    block_sum.s0 += (((bits4.s4 & 0xF000) >> 12) - 8) * shared_y.s3; \
    block_sum.s0 += ((bits4.s6 & 0x000F) - 8) * shared_y.s4; \
    block_sum.s0 += (((bits4.s6 & 0x00F0) >> 4) - 8) * shared_y.s5; \
    block_sum.s0 += (((bits4.s6 & 0x0F00) >> 8) - 8) * shared_y.s6; \
    block_sum.s0 += (((bits4.s6 & 0xF000) >> 12) - 8) * shared_y.s7; \
    block_sum.s1 += ((bits4.s5 & 0x000F) - 8) * shared_y.s0; \
    block_sum.s1 += (((bits4.s5 & 0x00F0) >> 4) - 8) * shared_y.s1; \
    block_sum.s1 += (((bits4.s5 & 0x0F00) >> 8) - 8) * shared_y.s2; \
    block_sum.s1 += (((bits4.s5 & 0xF000) >> 12) - 8) * shared_y.s3; \
    block_sum.s1 += ((bits4.s7 & 0x000F) - 8) * shared_y.s4; \
    block_sum.s1 += (((bits4.s7 & 0x00F0) >> 4) - 8) * shared_y.s5; \
    block_sum.s1 += (((bits4.s7 & 0x0F00) >> 8) - 8) * shared_y.s6; \
    block_sum.s1 += (((bits4.s7 & 0xF000) >> 12) - 8) * shared_y.s7; \

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
__kernel void kernel_gemv_noshuffle(
        __read_only  image1d_buffer_t src0_q,  // quantized A
        global half2  * src0_d,  // A scales
        global const float * src1,    // B
        ulong offset1,            // offset to B (0)
        global float * dst,     // C
        ulong offsetd,            // offset to C (0)
        uint K,               // K
        int ne01,               // M
        int ne02,               // 1
        int ne10,               // K
        int ne12,               // 1
        int ne0,                // M
        int ne1,                // N
        int r2,                 // 1
        int r3)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid    = get_sub_group_local_id();

    __private uint4     regA;
    __private half2     regS;
    __private float8    regB;

    __private float2 totalSum = (float2)(0.0f);
    global const float * src1_f = (global const float *)((global const char *)src1 + offset1);

    // loop along K in block granularity, skip 4 blocks every iter
#ifndef K_BLOCKS
#define K_BLOCKS (K / QK4_0)
#endif
    for (uint k = groupId; k < K_BLOCKS; k += N_SIMDGROUP) {
        regS = src0_d[gid + k * LINE_STRIDE_A]; // each fiber loads scale of two rows
        // first 4 fibers in each wave load 8 B values to its private scope
        if (slid < 4) {
            regB.s0123 = vload4(slid * 2 + k * 8, src1_f);
            regB.s4567 = vload4(1 + slid * 2 + k * 8, src1_f);
        }

        // load half weights for two blocks in consecutive rows
        regA.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
        regA.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
        regA.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
        regA.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAT
        dequantizeBlockAccum_ns_sgbroadcast_8_hi(totalSum, as_ushort8(regA), regS, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_hi(totalSum, as_ushort8(regA), regS, regB);
#endif // VECTOR_SUB_GROUP_BROADCAT

        regA.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;
        regA.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;
        regA.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;
        regA.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAT
        dequantizeBlockAccum_ns_sgbroadcast_8_lo(totalSum, as_ushort8(regA), regS, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_lo(totalSum, as_ushort8(regA), regS, regB);
#endif // VECTOR_SUB_GROUP_BROADCAT
    }

    __local float2 reduceLM[SIMDGROUP_WIDTH * (N_SIMDGROUP - 1)];
    if (groupId > 0) {
        reduceLM[SIMDGROUP_WIDTH * (groupId - 1) + slid] = totalSum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0) {
        for (uint i = 0; i < N_SIMDGROUP - 1; ++i) {
            totalSum += reduceLM[SIMDGROUP_WIDTH * i + slid];
        }
    }

    // 2 outputs per fiber in wave 0
    if (groupId == 0) {
        dst = (global float*)((global char*)dst + offsetd);
        vstore2(totalSum, 0, &(dst[gid * 2]));
    }

}
