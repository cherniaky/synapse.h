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