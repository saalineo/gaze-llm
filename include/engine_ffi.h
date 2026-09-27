#ifndef ENGINE_FFI_H
#define ENGINE_FFI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void* data_ptr;
    size_t num_elements;
    size_t element_size_bytes;
    int32_t dtype; // 0=FP32, 1=FP16, 2=BF16, 3=INT8
} MojoTensorBuffer;

typedef struct {
    int32_t* block_table_ptr;
    size_t num_blocks;
    size_t block_size;
} MojoBlockTable;

typedef struct {
    int32_t status_code; // 0 = Success, non-zero = Error
    const char* error_message;
} FfiResult;

#ifdef __cplusplus
}
#endif

#endif // ENGINE_FFI_H
