#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <iostream>
#include <sstream>
#include <string>

#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////////////////////
#define CHECK_ERR(x) do {check_err((x), #x, __FILE__, __LINE__);} while (false)
inline void check_err(cudaError_t err, const char *func, const char *file, int line) {
    if (err) {
        std::cerr << "(CUDA) " << file << ": " << line << "\ncudaError_T: "
            << cudaGetErrorName(err) << " (" << cudaGetErrorString(err) << ") \n\""
            << func << "\"" << std::endl;
        exit(err);
    }
}

inline void check_err(bool err, const char *func, const char *file, int line) {
    if (err) {
        std::cerr << "(CUDA) " << file << ": " << line << "\n\"" << func << "\"" << std::endl;
        exit(-1);
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////

#endif //CUDA_UTILS_CUH
