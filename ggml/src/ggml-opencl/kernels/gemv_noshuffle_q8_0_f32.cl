#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#endif

#define QK8_0 32
#define N_SIMDGROUP 16

#define dequantizeBlockAccum_ns_sgbroadcast_1(total_sums, bits8, scale, y) \
    float shared_y; \
    char elem; \
    float block_sum = 0.0f; \
                                             \
    shared_y = sub_group_broadcast(y.s0, 0); \
    elem = (char)(bits8.s0 & 0x000000FF); \
    block_sum += convert_int(elem) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 0); \
    elem = (char)((bits8.s0 & 0x0000FF00) >> 8); \
    block_sum += convert_int(elem) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 0); \
    elem = (char)((bits8.s0 & 0x00FF0000) >> 16); \
    block_sum += convert_int(elem) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 0); \
    elem = (char)((bits8.s0 & 0xFF000000) >> 24); \
    block_sum += convert_int(elem) * shared_y; \
                                             \
    shared_y = sub_group_broadcast(y.s4, 0); \
    elem = (char)(bits8.s1 & 0x000000FF); \
    block_sum += convert_int(elem) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 0); \
    elem = (char)((bits8.s1 & 0x0000FF00) >> 8); \
    block_sum += convert_int(elem) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 0); \
    elem = (char)((bits8.s1 & 0x00FF0000) >> 16); \
    block_sum += convert_int(elem) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 0); \
    elem = (char)((bits8.s1 & 0xFF000000) >> 24); \
    block_sum += convert_int(elem) * shared_y; \
                                             \
    shared_y = sub_group_broadcast(y.s0, 1); \
    elem = (char)(bits8.s2 & 0x000000FF); \
    block_sum += convert_int(elem) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 1); \
    elem = (char)((bits8.s2 & 0x0000FF00) >> 8); \
    block_sum += convert_int(elem) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 1); \
    elem = (char)((bits8.s2 & 0x00FF0000) >> 16); \
    block_sum += convert_int(elem) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 1); \
    elem = (char)((bits8.s2 & 0xFF000000) >> 24); \
    block_sum += convert_int(elem) * shared_y; \
                                             \
    shared_y = sub_group_broadcast(y.s4, 1); \
    elem = (char)(bits8.s3 & 0x000000FF); \
    block_sum += convert_int(elem) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 1); \
    elem = (char)((bits8.s3 & 0x0000FF00) >> 8); \
    block_sum += convert_int(elem) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 1); \
    elem = (char)((bits8.s3 & 0x00FF0000) >> 16); \
    block_sum += convert_int(elem) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 1); \
    elem = (char)((bits8.s3 & 0xFF000000) >> 24); \
    block_sum += convert_int(elem) * shared_y; \
                                             \
    shared_y = sub_group_broadcast(y.s0, 2); \
    elem = (char)(bits8.s4 & 0x000000FF); \
    block_sum += convert_int(elem) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 2); \
    elem = (char)((bits8.s4 & 0x0000FF00) >> 8); \
    block_sum += convert_int(elem) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 2); \
    elem = (char)((bits8.s4 & 0x00FF0000) >> 16); \
    block_sum += convert_int(elem) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 2); \
    elem = (char)((bits8.s4 & 0xFF000000) >> 24); \
    block_sum += convert_int(elem) * shared_y; \
                                             \
    shared_y = sub_group_broadcast(y.s4, 2); \
    elem = (char)(bits8.s5 & 0x000000FF); \
    block_sum += convert_int(elem) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 2); \
    elem = (char)((bits8.s5 & 0x0000FF00) >> 8); \
    block_sum += convert_int(elem) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 2); \
    elem = (char)((bits8.s5 & 0x00FF0000) >> 16); \
    block_sum += convert_int(elem) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 2); \
    elem = (char)((bits8.s5 & 0xFF000000) >> 24); \
    block_sum += convert_int(elem) * shared_y; \
                                             \
    shared_y = sub_group_broadcast(y.s0, 3); \
    elem = (char)(bits8.s6 & 0x000000FF); \
    block_sum += convert_int(elem) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 3); \
    elem = (char)((bits8.s6 & 0x0000FF00) >> 8); \
    block_sum += convert_int(elem) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 3); \
    elem = (char)((bits8.s6 & 0x00FF0000) >> 16); \
    block_sum += convert_int(elem) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 3); \
    elem = (char)((bits8.s6 & 0xFF000000) >> 24); \
    block_sum += convert_int(elem) * shared_y; \
                                             \
    shared_y = sub_group_broadcast(y.s4, 3); \
    elem = (char)(bits8.s7 & 0x000000FF); \
    block_sum += convert_int(elem) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 3); \
    elem = (char)((bits8.s7 & 0x0000FF00) >> 8); \
    block_sum += convert_int(elem) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 3); \
    elem = (char)((bits8.s7 & 0x00FF0000) >> 16); \
    block_sum += convert_int(elem) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 3); \
    elem = (char)((bits8.s7 & 0xFF000000) >> 24); \
    block_sum += convert_int(elem) * shared_y; \
    total_sums += block_sum * scale; \

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
__kernel void kernel_gemv_noshuffle_q8_0_f32(
        __read_only  image1d_buffer_t src0_q,  // quantized A
        global half  * src0_d,  // A scales
        __read_only  image1d_buffer_t src1,    // B
        ulong offset1,            // offset to B (0)
        global float * dst,     // C
        ulong offsetd,            // offset to C
        int ne00,               // K
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

    uint K = ne00;
    uint M = ne01;

#ifndef LINE_STRIDE_A
#define LINE_STRIDE_A M
#endif
#ifndef BLOCK_STRIDE_A
#define BLOCK_STRIDE_A (8 * M)
#endif
#ifndef K_BLOCKS
#define K_BLOCKS (K / QK8_0)
#endif

    __private uint8     regA;
    __private half      regS;
    __private float8    regB;

    __private float totalSum = (float)(0.0f);

    // loop along K in block granularity, skip 4 blocks every iter
    #pragma unroll 1 /* tell compiler not to unroll */
    for (uint k = groupId; k < K_BLOCKS; k += N_SIMDGROUP) {
        regS = src0_d[gid + k * LINE_STRIDE_A]; // each fiber loads scale of one rows
        // first 4 fibers in each wave load 8 B values to its private scope
        if (slid < 4) {
            regB.s0123 = read_imagef(src1, (slid * 2 + k * 8));
            regB.s4567 = read_imagef(src1, (1 + slid * 2 + k * 8));
        }

        // load weights for one block in consecutive rows
        regA.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
        regA.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
        regA.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
        regA.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;
        regA.s4 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;
        regA.s5 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;
        regA.s6 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;
        regA.s7 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;

        dequantizeBlockAccum_ns_sgbroadcast_1(totalSum, regA, regS, regB);
    }

    __local float reduceLM[SIMDGROUP_WIDTH * (N_SIMDGROUP - 1)];
    if (groupId > 0) {
        reduceLM[SIMDGROUP_WIDTH * (groupId - 1) + slid] = totalSum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0) {
        for (uint i = 0; i < N_SIMDGROUP - 1; ++i) {
            totalSum += reduceLM[SIMDGROUP_WIDTH * i + slid];
        }
    }

    // 1 outputs per fiber in wave 0
    if (groupId == 0) {
        dst = (global float*)((global char*)dst + offsetd);
        dst[gid] = totalSum;
    }
}
