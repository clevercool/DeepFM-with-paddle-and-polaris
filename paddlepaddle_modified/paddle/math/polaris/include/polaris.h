/**
 * @file      polaris.h
 *
 * @brief     A FPGA-accelerated library of primitives for deep neural networks. \n
 *            Provides highly tuned implementations of routines arising frequently
 *            in deep learning applications.
 *
 * @authors   isa@baidu.com
 *
 * @copyright (C) 2017 Baidu, Inc
 */

#ifndef BAIDU_POLARIS_INCLUDE_POLARIS_H
#define BAIDU_POLARIS_INCLUDE_POLARIS_H

#include <stdint.h>
#include <string.h>

/**
 * @defgroup datatype Data Types
 * All data types and enums of Polaris library.
 */

/**
 * @defgroup context Context Management
 * Functions related to operations of PolarisContext.
 */

/**
 * @defgroup memory Memory Management
 * Functions related to memory allocation, free and copy, including memcpy between CPU and FPGA
 * and kinds of on-FPGA memcpy.
 */

/**
 * @defgroup compute Computation Interface
 * Functions that involve computation. This group provides the core computation ability of
 * Polaris library.
 */

/**
 * @defgroup query Device Management
 * Functions related to device management, including quering the version and matching devices.
 */

#include "polaris/driver/types.h"

/**
 * @ingroup datatype
 * @brief Context object
 *
 * The data structure that holds polaris library context
 */
typedef struct {
    /// Device id
    int devid;

    /// file descriptor to the device
    int fd;
} PolarisContext;

/**
 * @ingroup datatype
 * Return status of polaris interface
 */
typedef enum {
    /// Everything goes fine
    POLARIS_OK = 0,

    /// The requested operation is not supported
    POLARIS_ERR_NOT_SUPPORT = 1,

    /// Invalid paramaters
    POLARIS_ERR_INVALID = 2,

    /// Runtime error
    POLARIS_ERR_RUNTIME = 3,
} PolarisStatus;

/**
 * @ingroup datatype
 * Function types supported by polaris_elementwise()
 */
typedef enum {
    /// element-wise addition
    POLARIS_ADD = POLARIS_ELEMENTWISE_AXPY,

    /// element-wise multiplication
    POLARIS_MUL = POLARIS_ELEMENTWISE_MUL,
} PolarisElementWiseType;

/**
 * @ingroup context
 * @brief Create a polaris context object on a specific device.
 *
 * @param[in] devid   Device id
 *
 * @return Pointer to a PolarisContext object, NULL for failure
 *
 * @see     polaris_get_devices()
 */
PolarisContext* polaris_create_context(int devid);

/**
 * @ingroup context
 * @brief Free and destroy a polaris context
 *
 * @param[in] ctxt   Pointer to a PolarisContext object
 */
void polaris_destroy_context(PolarisContext* ctxt);

/**
 * @ingroup memory
 * @brief Allocate a block of memory on FPGA.
 *
 * The allocated memory block is associated with the Context object, it will be auto
 * released if the context is destroyed. But there is no boundary in the access of memory
 * allocated in different contexts, that is to say, one context could read/write the
 * content of the memory allocated in another context.
 *
 * @param[in]   ctxt   pointer to PolarisContext object
 * @param[in]   size   size of memory in bytes to allocate
 * @param[out]  ptr    malloced FPGA memory, NULL on error
 *
 * @return Execution status, #POLARIS_OK on success.
 */
PolarisStatus polaris_malloc(PolarisContext* ctxt, size_t size, void** ptr);

/**
 * @ingroup memory
 * @brief Free a block of memory on FPGA.
 *
 * You can only free those memories that were allocated in the same context.
 *
 * @param[in] ctxt  pointer to PolarisContext object
 * @param[in] ptr   start address of the FPGA memory to be freed
 *
 * @return Execution status, #POLARIS_OK on success.
 */
PolarisStatus polaris_free(PolarisContext* ctxt, void* ptr);

/**
 * @ingroup memory
 * @brief Copy block of memory between CPU and FPGA.
 *
 * Copy @p size bytes from the memory area pointed to by @p src to the memory area pointed to
 * by @p dest, where @p kind specifies the direction of the copy, and must by one of
 * #POLARIS_DEVICE_TO_HOST, #POLARIS_HOST_TO_DEVICE, #POLARIS_DEVICE_TO_DEVICE
 *
 * @param[in]  ctxt    pointer to PolarisContext object
 * @param[in]  kind    direction of the transfer
 * @param[out] dest    destination FPGA/CPU memory address
 * @param[in]  src     source CPU/FPGA memory address
 * @param[in]  size    number of bytes to be copied
 *
 * @return Execution status, #POLARIS_OK on success.
 *
 * @see PolarisMemcpyKind
 */
PolarisStatus polaris_memcpy(PolarisContext* ctxt, PolarisMemcpyKind kind,
                             void* dest, const void* src, size_t size);
/**
 * @ingroup compute
 * @brief Perform matrix multiplication with bias and activation support.
 *
 * This function perform the calculation as the formula:
 *
 *     c = activation( alpha * op(a) * op(b) + beta * c + bias )
 *
 * This function behaves similar with `SGEMM` function in
 * [CBLAS](http://www.netlib.org/lapack/explore-html/d4/de2/sgemm_8f.html), with the extended
 * ability of adding bias vector and performing activation functions, which match better for the
 * need of deep neural network. In another word, this function provides the ability of the forward
 * process of a full-connected layer.
 *
 * Parameters like @p trans_a, @p trans_b, @p m, @p n, @p k, @p alpha, @p a, @p lda @p b, @p ldb,
 * @p beta, @p c, @p ldc all have the same meaning with that in CBLAS `SGEMM`.
 *
 * @p bias is a vector with dimension `1 * n`, which will be added to each row of output
 * matrix @p c.
 *
 * @p activation indicates the activation function type to be performed after the bias procession.
 *
 * @note this function best performed with @p trans_a equals to #POLARIS_NO_TRANS and @p trans_b
 * equals to #POLARIS_TRANS. (Currently supports calling with this mode only)
 *
 *
 * @param[in]     ctxt         pointer to PolarisContext object
 * @param[in]     trans_a      if #POLARIS_NO_TRANS, @p a is `m * k`, otherwise `k * m`
 * @param[in]     trans_b      if #POLARIS_NO_TRANS, @p b is `n * k`, otherwise `k * n`
 * @param[in]     m            dimension m
 * @param[in]     n            dimension n
 * @param[in]     k            dimension k
 * @param[in]     a            FPGA address of matrix a
 * @param[in]     b            FPGA address of matrix b
 * @param[in,out] c            FPGA address of matrix c
 * @param[in]     bias         FPGA address of bias
 *
 * @return Execution status, #POLARIS_OK on success.
 */
PolarisStatus polaris_gemm(PolarisContext* ctxt,
                           int m, int n, int k,
                           const float* a,
                           const float* b,
                           void* c,
                           const float* bias);

/**
 * @ingroup memory
 * @brief Fill a range of FPGA memory with zero.
 *
 * This function fills the first @p size bytes of the FPGA memory area pointed to by @p ptr with 0.
 *
 * @param[in] ctxt   pointer to PolarisContext object
 * @param[in] ptr    address of the FPGA memory to be set
 * @param[in] size   size of the memory (in bytes)
 *
 * @return Execution status, #POLARIS_OK on success.
 */
PolarisStatus polaris_memset(PolarisContext* ctxt, void* ptr, size_t size);

/**
 * @ingroup compute
 * @brief Compute activation functions.
 *
 * This function calculate the following formula:
 *
 *     b[i] = alpha * ActivationType(a[i]) + beta * b[i]
 *
 * where both @p a and @p b have @p length elements. Activation type is indicated by @p type.
 *
 * @param[in]   ctxt     pointer to PolarisContext object
 * @param[in]   type     activation type
 * @param[in]   length   elemnet numbers of a, b
 * @param[in]   alpha    alpha value of a
 * @param[in]   a        input matrix
 * @param[in]   beta     beta value of b
 * @param[out]  b        output matrix
 *
 * @return Execution status, #POLARIS_OK on success.
 */
PolarisStatus polaris_activation(PolarisContext* ctxt, PolarisActivationType type, size_t length,
                                 float alpha, const float* a, float beta, float* b);

/**
 * @ingroup compute
 * @brief Elementwise functions.
 *
 * Perform element-wise operations indicated by the following fomula:
 *
 *     c[i] = ElementWiseOperation(alpha0 * a[i], alpha1 * b[i]) + beta * c[i]
 *
 * where @p type indicates the exact operation to be performed.
 *
 * - #POLARIS_ADD `:= (alpha0 * a[i]) + (alpha1 * b[i])`
 * - #POLARIS_MUL `:= (alpha0 * a[i]) * (alpha1 * b[i])`
 *
 * @param[in]  ctxt         pointer to PolarisContext
 * @param[in]  type         element wise operation type
 * @param[in]  length       elemnet numbers of a, b, c
 * @param[in]  alpha0       scalar parameter of a
 * @param[in]  a            input matrix
 * @param[in]  alpha1       scalar parameter of b
 * @param[in]  b            input matrix
 * @param[in]  beta         scalar parameter of c
 * @param[out] c            output matrix
 *
 * @return Execution status, #POLARIS_OK on success.
 */
PolarisStatus polaris_elementwise(PolarisContext* ctxt, PolarisElementWiseType type, size_t length,
                                    float alpha0, const float* a,
                                    float alpha1, const float* b,
                                    float beta, float* c);

/**
 * @ingroup memory
 * @brief Perform batched memcpy in a 2-D pattern. Often used to concat.
 *
 * Copy @p m rows of data, the size (in bytes) of each row is indicated by
 * @p n, and the source and destination address of i-th copy
 * (i = 0, 1, ..., @p m - 1) is `&src[i * stride_src]` and
 * `&dest[i * stride_dest]` separately.
 *
 * @param[in]  ctxt          pointer to PolarisContext
 * @param[in]  kind          direction of the transfer
 * @param[in]  m             rows of matrix data (number of memory blocks to be copied)
 * @param[in]  n             cols (in `bytes`) of matrix data (size of each memory copy)
 * @param[out] dest          destination memory address
 * @param[in]  stride_dest   destination step (in `bytes`)
 * @param[in]  src           source memory address
 * @param[in]  stride_src    source step (in `types`)
 *
 * @return Execution status, #POLARIS_OK on success.
 */
PolarisStatus polaris_memcpy_2d(PolarisContext *ctxt,
                                PolarisMemcpyKind kind,
                                int m, int n,
                                void *dest, int stride_dest,
                                const void *src, int stride_src);

#endif
