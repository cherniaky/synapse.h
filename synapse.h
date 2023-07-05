#ifndef SYNAPSE_H_
#define SYNAPSE_H_

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef S_CALLOC
#define S_CALLOC calloc
#endif // S_CALLOC

#ifndef S_ASSERT
#include <assert.h>
#define S_ASSERT assert
#endif // S_ASSERT

typedef struct
{
    size_t rows;
    size_t cols;
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i) * (m).cols + (j)]

Mat mat_alloc(size_t rows, size_t cols);
void mat_dot(Mat dist, Mat a, Mat b);
void mat_sum(Mat dist, Mat a, Mat b);
void mat_print(Mat a);
#endif // SYNAPSE_H_

#ifdef SYNAPSE_IMPLEMENTATION

Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.es = S_CALLOC(rows * cols, sizeof(*m.es));
    S_ASSERT(m.es != NULL);
    return m;
};

void mat_dot(Mat dist, Mat a, Mat b)
{
    (void)dist;
    (void)a;
    (void)b;
}
void mat_sum(Mat dist, Mat a, Mat b)
{
    (void)dist;
    (void)a;
    (void)b;
}

void mat_print(Mat m)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            printf("%f", MAT_AT(m, i, j));
        }
    }
    printf("\n");
}

#endif // SYNAPSE_IMPLEMENTATION
