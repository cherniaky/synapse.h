#include <time.h>

#define SYNAPSE_IMPLEMENTATION ;
#include "synapse.h"

int main()
{
    srand(time(0));
    Mat a = mat_alloc(2, 2);
    mat_rand(a, 1, 2);
    mat_print(a);

    Mat b = mat_alloc(2, 2);
    mat_rand(b, 1, 2);
    mat_print(b);

    mat_sum(a, b);
    mat_print(a);
    return 0;
}