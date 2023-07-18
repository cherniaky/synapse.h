#include <stdio.h>
#include <assert.h>

#define STB_IMAGE_IMPLEMENTATION
#include "dev_deps/stb_image.h"

#define SYNAPSE_IMPLEMENTATION
#include "synapse.h"

char *args_shift(int *argc, char ***argv)
{
    assert(*argc > 0);
    char *result = **argv;
    (*argc)--;
    (*argv)++;
    return result;
}

int main(int argc, char **argv)
{
    args_shift(&argc, &argv);
    if (argc < 0)
    {
        fprintf(stderr, "ERROR: no input file was provided\n");
        return 1;
    }

    const char *img_file_path = args_shift(&argc, &argv);

    int img_width, img_height, n;
    uint8_t *img_pixels = (uint8_t *)stbi_load(img_file_path, &img_width, &img_height, &n, 0);
    if (img_pixels == NULL)
    {
        fprintf(stderr, "Could not read image %s\n", img_file_path);
        return 1;
    }
    if (n != 1)
    {
        fprintf(stderr, "Image %s is %d bits image. Only 8 bit grayscale images is supported\n", img_file_path, 8 * n);
        return 1;
    }
    fprintf(stdout, "%s is %dx%d %d bits\n", img_file_path, img_width, img_height, n * 8);

    Mat td = mat_alloc(img_width * img_height, 3);

    for (size_t y = 0; y < img_height; y++)
    {
        for (size_t x = 0; x < img_width; x++)
        {
            float nx = (float)x / (img_width - 1);
            float ny = (float)y / (img_height - 1);
            size_t i = y * img_width + x;
            float nv = img_pixels[i] / 255.f;

            MAT_AT(td, i, 0) = nx;
            MAT_AT(td, i, 1) = ny;
            MAT_AT(td, i, 2) = nv;
        }
    }

    // MAT_PRINT(td);
    // const char *out_file_path = "img.mat";
    // FILE *out = fopen(out_file_path, "wb");
    // if (out == NULL)
    // {
    //     fprintf(stderr, "Could not open file %s\n", out_file_path);
    //     return 1;
    // }
    // mat_save(out, td);

    // printf("Generated %s from %s\n", out_file_path, img_file_path);
    Mat ti = {
        .rows = td.rows,
        .cols = 2,
        .stride = td.stride,
        .es = &MAT_AT(td, 0, 0),
    };

    Mat to = {
        .rows = td.rows,
        .cols = 1,
        .stride = td.stride,
        .es = &MAT_AT(td, 0, 2),
    };
    
    size_t arch[] = {2, 28, 1};

    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, -4, 4);
    float rate = 1.f;

    for (size_t epochs = 0; epochs < 10000; epochs++)
    {
        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, rate);
        if (epochs % 100 == 0)
        {
            printf("%zu: cost = %f\n", epochs, nn_cost(nn, ti, to));
        }
    }

    return 0;
}