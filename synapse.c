#include <time.h>

#define SYNAPSE_IMPLEMENTATION ;
#include "synapse.h"

typedef struct
{
    Mat a0;
    Mat w1, b1, a1;
    Mat w2, b2, a2;
} XOR;

float forward_xor(XOR m, float x1, float x2)
{
    MAT_AT(m.a0, 0, 0) = x1;
    MAT_AT(m.a0, 0, 1) = x2;

    mat_dot(m.a1, m.a0, m.w1);
    mat_sum(m.a1, m.b1);
    mat_sig(m.a1);

    mat_dot(m.a2, m.a1, m.w2);
    mat_sum(m.a2, m.b2);
    mat_sig(m.a2);

    return *m.a2.es;
}
int main()
{
    srand(time(0));

    XOR m;

    m.a0 = mat_alloc(1, 2);
    m.w1 = mat_alloc(2, 2);
    m.b1 = mat_alloc(1, 2);
    m.a1 = mat_alloc(1, 2);
    m.w2 = mat_alloc(2, 1);
    m.b2 = mat_alloc(1, 1);
    m.a2 = mat_alloc(1, 1);

    mat_rand(m.w1, -1, 1);
    mat_rand(m.b1, -1, 1);
    mat_rand(m.w2, -1, 1);
    mat_rand(m.b2, -1, 1);

    return 0;
}