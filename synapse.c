#include <time.h>

#define SYNAPSE_IMPLEMENTATION ;
#include "synapse.h"

int main()
{
    srand(time(0));

    Mat w1 = mat_alloc(2, 2);
    Mat b1 = mat_alloc(1, 2);
    Mat w2 = mat_alloc(2, 1);
    Mat b2 = mat_alloc(1, 1);

    mat_rand(w1, -1, 1);
    mat_rand(b1, -1, 1);
    mat_rand(w2, -1, 1);
    mat_rand(b2, -1, 1);

    MAT_PRINT(w1);
    MAT_PRINT(b1);
    MAT_PRINT(w2);
    MAT_PRINT(b2);

    return 0;
}