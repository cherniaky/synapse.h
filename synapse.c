#include <time.h>

#define SYNAPSE_IMPLEMENTATION ;
#include "synapse.h"

int main()
{
    srand(time(0));
    Mat a = mat_alloc(2, 3);
    mat_fill(a, 1);
    mat_print(a);
    printf("-------\n");
    Mat b = mat_alloc(3, 2);
    mat_fill(b, 1);
    mat_print(b);
    printf("-------\n");
    Mat dist = mat_alloc(2, 2);
    mat_dot(dist, a, b);
    mat_print(dist);
    return 0;
}