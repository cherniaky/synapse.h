#include <time.h>

#define SYNAPSE_IMPLEMENTATION ;
#include "synapse.h"

int main()
{
    srand(time(0));

    Mat x = mat_alloc(1, 2);

    Mat w1 = mat_alloc(2, 2);
    Mat b1 = mat_alloc(1, 2);
    Mat a1 = mat_alloc(1, 2);

    Mat w2 = mat_alloc(2, 1);
    Mat b2 = mat_alloc(1, 1);
    Mat a2 = mat_alloc(1, 1);

    mat_rand(w1, -1, 1);
    mat_rand(b1, -1, 1);
    mat_rand(w2, -1, 1);
    mat_rand(b2, -1, 1);

    MAT_AT(x, 0, 0) = 0;
    MAT_AT(x, 0, 1) = 1;

    mat_dot(a1, x, w1);
    mat_sum(a1, b1);
    mat_sig(a1);

    mat_dot(a2, a1, w2);
    mat_sum(a2, b2);
    mat_sig(a2);

    MAT_PRINT(a2);

    return 0;
}