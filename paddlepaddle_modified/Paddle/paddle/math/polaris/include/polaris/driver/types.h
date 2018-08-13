#ifndef BAIDU_POLARIS_DRIVER_INCLUDE_FPGA_DRIVER_TYPES_H
#define BAIDU_POLARIS_DRIVER_INCLUDE_FPGA_DRIVER_TYPES_H

#define FPGACTL_NODE_DEVNAME "fpgactl"

/// @addtogroup datatype
/// @{

/**
 * Indicates the copy direction
 */
typedef enum {
    /// Copy from FPGA to CPU
    POLARIS_DEVICE_TO_HOST = 0,

    /// Copy from CPU to FPGA
    POLARIS_HOST_TO_DEVICE = 1,

    /// Copy from FPGA to CPU
    POLARIS_DEVICE_TO_DEVICE = 2,
} PolarisMemcpyKind;

/**
 * Indicates whether transpose operation needs to be performed
 */
typedef enum {
    /// Non-transpose operation
    POLARIS_NO_TRANS = 0,

    /// Transpose operation
    POLARIS_TRANS = 1
} PolarisTransType;

/**
 * Types to specify the data precision
 */
typedef enum {
    /// 32-bit floating-point
    POLARIS_FP32 = 0,

    /// 16-bit floating-point
    POLARIS_FP16,

    /// 32-bit signed integer
    POLARIS_INT32,

    /// 16-bit signed integer
    POLARIS_INT16,

    /// 8-bit signed integer
    POLARIS_INT8
} PolarisDataType;

/**
 * Data format supported by convolution operation
 */
typedef enum {
    /// NHWC format
    POLARIS_FORMAT_NHWC = 0,

    /// NCHW format
    POLARIS_FORMAT_NCHW = 1,
} PolarisDataFormat;

/**
 * Function types supported by pooling operation
 */
typedef enum {
    /// Max-pooling
    POLARIS_POOLING_MAX = 0,

    /// Avg-pooling
    POLARIS_POOLING_AVG = 1,
} PolarisPoolingMode;

/// @cond IGNORED

/**
 * Function types supported by elementwise operation
 */
typedef enum {
    /// Illegal type
    POLARIS_ELEMENTWISE_ILLEGAL = 0,

    /// Element-wise softsign activation
    POLARIS_ELEMENTWISE_SOFTSIGN,

    /// Element-wise softsign deactivation
    POLARIS_ELEMENTWISE_DSOFTSIGN,

    /// Vector-vector operation: y := a*x + y
    POLARIS_ELEMENTWISE_AXPY,

    /// Element-wise tanh activation
    POLARIS_ELEMENTWISE_TANH,

    /// Element-wise tanh deactivation
    POLARIS_ELEMENTWISE_DTANH,

    /// Element-wise vsum
    POLARIS_ELEMENTWISE_VSUM,

    /// Element-wise memset
    POLARIS_ELEMENTWISE_MEMSET,

    /// Element-wise memcpy
    POLARIS_ELEMENTWISE_MEMCPY,

    /// Element-wise sigmoid activation
    POLARIS_ELEMENTWISE_SIGMOID,

    /// Element-wise relu activation
    POLARIS_ELEMENTWISE_RELU,

    /// Element-wise multiply
    POLARIS_ELEMENTWISE_MUL,

    /// Element-wise minimal
    POLARIS_ELEMENTWISE_MIN,

    /// Element-wise maximal
    POLARIS_ELEMENTWISE_MAX,
} PolarisElementwiseFunctionType;

/// @endcond

/**
 * Activation types
 */
typedef enum {
    /// Non-activation
    POLARIS_NO_ACTIVATION = 0,

    /// Tanh activation
    POLARIS_TANH = POLARIS_ELEMENTWISE_TANH,

    /// Sigmoid activation
    POLARIS_SIGMOID = POLARIS_ELEMENTWISE_SIGMOID,

    /// Relu activation
    POLARIS_RELU = POLARIS_ELEMENTWISE_RELU,

    /// Softsign activation
    POLARIS_SOFTSIGN = POLARIS_ELEMENTWISE_SOFTSIGN,
} PolarisActivationType;

/**
 * Conv-stream types
 */
typedef enum {
    /// Convmtx operation
    POLARIS_CONVSTREAM_CONVMTX1D = 0,

    /// Deconvmtx operation
    POLARIS_CONVSTREAM_DCONVMTX1D,

    /// maxpooling operation
    POLARIS_CONVSTREAM_MAXPOLLING1D,

    /// dmaxpooling operation
    POLARIS_CONVSTREAM_DMAXPOLLING1D,
} PolarisConvStreamFunctionType;

/// @} group datatype

/// @cond IGNORED
/************
 * CONSTANTS
 ************/
enum {
    /// max device number
    MAX_DEVICE_COUNT = 64,

    /// max length of version string
    MAX_VERSION_STRING_LENGTH = 32,

    /// max length of firmware-name string
    MAX_FIRMWARE_NAME_LENGTH = 32,
};
/// @endcond

#endif // include guard
