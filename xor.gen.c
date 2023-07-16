#define SYNAPSE_IMPLEMENTATION
#include "synapse.h"
#include <stdio.h>

int main()
{
    float td[] = {
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        0,
    };
    // size_t stride = 3;
    // size_t n = sizeof(td) / sizeof(td[0]) / stride;
    // Mat ti = {
    //     .rows = n,
    //     .cols = 2,
    //     .stride = stride,
    //     .es = td};

    // Mat to = {
    //     .rows = n,
    //     .cols = 1,
    //     .stride = stride,
    //     .es = td + 2};
    Mat data = {
        .cols = 3,
        .rows = 4,
        .es = td,
        .stride = 3};
    FILE *out = fopen("xor.mat", "wb");
    mat_save(out, data);
    fclose(out);

    FILE *in = fopen("xor.mat", "rb");
    Mat m = mat_load(in);
    MAT_PRINT(m);
    fclose(in);
    return 0;
}