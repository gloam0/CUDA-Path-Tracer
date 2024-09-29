#ifndef FRAME_TO_IMAGE_CUH
#define FRAME_TO_IMAGE_CUH

#include <common.cuh>
#include <vector>

#include <jpeglib.h>

#include "common.hpp"
#include "cuda_utils.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////
/// Save current device uchar4 image data as a JPG.
/// @param d_image_data Pointer to device uchar4 image data array.
/// @param quality JPG compression quality [0, 100].
/// @param filename File to write JPEG data to.
inline void save_frame_as_image(uchar4* d_image_data, int quality, const char* filename) {
    /* Copy device image data to host */
    size_t num_bytes = img::w * img::h * sizeof(uchar4);
    std::vector<unsigned char> h_image_data(num_bytes);
    CHECK_ERR(cudaMemcpy(h_image_data.data(), d_image_data, num_bytes, cudaMemcpyDeviceToHost));

    jpeg_compress_struct cinfo;
    jpeg_error_mgr jerr;
    JSAMPROW row_pointer[1];

    /* Create JPEG compression object */
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    /* Destination */
    FILE* outfile = fopen(filename, "wb");
    if (!outfile) {
        std::cerr << "save_frame_as_image(): Error opening outfile." << std::endl;
        return;
    }

    jpeg_stdio_dest(&cinfo, outfile);

    /* Compression params */
    cinfo.image_width = img::w;
    cinfo.image_height = img::h;
    cinfo.input_components = 4;
    cinfo.in_color_space = JCS_EXT_RGBA;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, true);

    /* Start compression */
    jpeg_start_compress(&cinfo, true);
    int row_stride = img::w * 4;
    while (cinfo.next_scanline < cinfo.image_height) {
        int src_row = cinfo.image_height - 1 - cinfo.next_scanline;
        row_pointer[0] = &h_image_data[src_row * row_stride];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    /* Finish and clean up */
    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);
}

#endif //FRAME_TO_IMAGE_CUH
