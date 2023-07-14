#ifndef SYNAPSE_H_
#define SYNAPSE_H_

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifndef S_CALLOC
#define S_CALLOC calloc
#endif // S_CALLOC

#ifndef S_ASSERT
#include <assert.h>
#define S_ASSERT assert
#endif // S_ASSERT

#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])

float rand_float(void);
float sigmoidf(float x);

typedef struct
{
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i) * (m).stride + (j)]

Mat mat_alloc(size_t rows, size_t cols);
void mat_save(FILE *out, Mat m);
void mat_load(FILE *in, Mat m);
void mat_dot(Mat dist, Mat a, Mat b);
void mat_sum(Mat dist, Mat a);
void mat_sig(Mat m);
void mat_print(Mat m, const char *name, size_t padding);
void mat_rand(Mat m, float low, float high);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dist, Mat src);
void mat_fill(Mat m, float x);
#define MAT_PRINT(m) mat_print(m, #m, 0)
Mat mat_t(Mat old);

typedef struct
{
    size_t count;
    Mat *ws, *bs, *as;
} NN;

#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_print(NN nn, char *name);
#define NN_PRINT(nn) nn_print((nn), #nn)
void nn_rand(NN nn, float low, float high);
void nn_forward(NN nn);
float nn_cost(NN nn, Mat ti, Mat to);
void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to);
void nn_backprop(NN nn, NN g, Mat ti, Mat to);
void nn_learn(NN nn, NN g, float rate);
void nn_zero(NN nn);

#endif // SYNAPSE_H_

#ifdef SYNAPSE_IMPLEMENTATION

float rand_float()
{
    return (float)rand() / (float)RAND_MAX;
}

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = S_CALLOC(rows * cols, sizeof(*m.es));
    S_ASSERT(m.es != NULL);
    return m;
}

void mat_save(FILE *out, Mat m){

}
void mat_load(FILE *in, Mat m)
{
    (void)in;
    (void)m;
}

void mat_dot(Mat dist, Mat a, Mat b)
{
    S_ASSERT(a.cols == b.rows);
    S_ASSERT(dist.rows == a.rows);
    S_ASSERT(dist.cols == b.cols);

    for (size_t i = 0; i < dist.rows; i++)
    {
        for (size_t j = 0; j < dist.cols; j++)
        {
            MAT_AT(dist, i, j) = 0;
            for (size_t k = 0; k < a.cols; k++)
            {
                MAT_AT(dist, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

void mat_sum(Mat dist, Mat a)
{
    S_ASSERT(dist.rows == a.rows);
    S_ASSERT(dist.cols == a.cols);
    for (size_t i = 0; i < dist.rows; i++)
    {
        for (size_t j = 0; j < dist.cols; j++)
        {
            MAT_AT(dist, i, j) += MAT_AT(a, i, j);
        }
    }
}

void mat_sig(Mat m)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}

void mat_print(Mat m, const char *name, size_t padding)
{

    printf("%*s%s = [\n", (int)padding, "", name);
    for (size_t i = 0; i < m.rows; i++)
    {
        printf("%*s", (int)padding, "");
        for (size_t j = 0; j < m.cols; j++)
        {
            printf("   %f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int)padding, "");
}

void mat_fill(Mat m, float x)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = x;
        }
    }
}

void mat_rand(Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = rand_float() * (high - low) + low;
        }
    }
}

Mat mat_row(Mat m, size_t row)
{
    return (Mat){
        .es = &MAT_AT(m, row, 0),
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
    };
}

void mat_copy(Mat dist, Mat src)
{
    S_ASSERT(dist.rows == src.rows);
    S_ASSERT(dist.cols == src.cols);

    for (size_t i = 0; i < dist.rows; i++)
    {
        for (size_t j = 0; j < dist.cols; j++)
        {
            MAT_AT(dist, i, j) = MAT_AT(src, i, j);
        }
    }
}
Mat mat_t(Mat old)
{
    Mat mat = mat_alloc(old.cols, old.rows);
    for (size_t i = 0; i < old.rows; i++)
    {
        for (size_t j = 0; j < old.cols; j++)
        {
            MAT_AT(mat, j, i) = MAT_AT(old, i, j);
        }
    }
    return mat;
}

NN nn_alloc(size_t *arch, size_t arch_count)
{
    S_ASSERT(arch_count > 0);

    NN nn;
    nn.count = arch_count - 1;

    nn.ws = S_CALLOC(nn.count, sizeof(*nn.ws));
    S_ASSERT(nn.ws != NULL);
    nn.bs = S_CALLOC(nn.count, sizeof(*nn.bs));
    S_ASSERT(nn.bs != NULL);
    nn.as = S_CALLOC(arch_count, sizeof(*nn.as));
    S_ASSERT(nn.as != NULL);

    nn.as[0] = mat_alloc(1, arch[0]);
    for (size_t i = 1; i < arch_count; i++)
    {
        nn.ws[i - 1] = mat_alloc(nn.as[i - 1].cols, arch[i]);
        nn.bs[i - 1] = mat_alloc(nn.as[i - 1].rows, arch[i]);
        nn.as[i] = mat_alloc(nn.as[i - 1].rows, arch[i]);
    }

    return nn;
}

void nn_print(NN nn, char *name)
{
    char buf[256];

    printf("%s = [\n", name);
    Mat *ws = nn.ws;
    Mat *bs = nn.bs;
    for (size_t i = 0; i < nn.count; i++)
    {
        snprintf(buf, sizeof(buf), "ws[%zu]", i);
        mat_print(ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs[%zu]", i);
        mat_print(bs[i], buf, 4);
    }

    printf("]\n");
}

void nn_rand(NN nn, float low, float high)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

void nn_forward(NN nn)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        mat_dot(nn.as[i + 1], nn.as[i], nn.ws[i]);
        mat_sum(nn.as[i + 1], nn.bs[i]);
        mat_sig(nn.as[i + 1]);
    }
}

float nn_cost(NN nn, Mat ti, Mat to)
{
    S_ASSERT(ti.rows == to.rows);
    S_ASSERT(to.cols == NN_OUTPUT(nn).cols);

    size_t n = ti.rows;

    float c = 0;
    for (size_t i = 0; i < n; i++)
    {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);

        mat_copy(NN_INPUT(nn), x);
        nn_forward(nn);

        size_t q = to.cols;
        for (size_t j = 0; j < q; j++)
        {
            float diff = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
            c += diff * diff;
        }
    }

    return c / (float)n;
}

void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to)
{
    float saved;
    float c = nn_cost(nn, ti, to);

    for (size_t i = 0; i < nn.count; i++)
    {
        for (size_t j = 0; j < nn.ws[i].rows; j++)
        {
            for (size_t k = 0; k < nn.ws[i].cols; k++)
            {
                saved = MAT_AT(nn.ws[i], j, k);
                MAT_AT(nn.ws[i], j, k) += eps;
                MAT_AT(g.ws[i], j, k) = (nn_cost(nn, ti, to) - c) / eps;
                MAT_AT(nn.ws[i], j, k) = saved;
            }
        }

        for (size_t j = 0; j < nn.bs[i].rows; j++)
        {
            for (size_t k = 0; k < nn.bs[i].cols; k++)
            {
                saved = MAT_AT(nn.bs[i], j, k);
                MAT_AT(nn.bs[i], j, k) += eps;
                MAT_AT(g.bs[i], j, k) = (nn_cost(nn, ti, to) - c) / eps;
                MAT_AT(nn.bs[i], j, k) = saved;
            }
        }
    }
}

void nn_backprop(NN nn, NN g, Mat ti, Mat to)
{
    S_ASSERT(ti.rows == to.rows);
    size_t n = ti.rows;
    S_ASSERT(NN_OUTPUT(nn).cols == to.cols);

    nn_zero(g);
    // i - current sample
    for (size_t i = 0; i < n; i++)
    {
        mat_copy(NN_INPUT(nn), mat_row(ti, i));
        nn_forward(nn);

        for (size_t j = 0; j <= nn.count; j++)
        {
            mat_fill(g.as[j], 0);
        }

        for (size_t j = 0; j < to.cols; j++)
        {
            MAT_AT(NN_OUTPUT(g), 0, j) = 2 * (MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(to, i, j));
        }

        for (int l = nn.count - 1; l >= 0; l--)
        {
            for (size_t m = 0; m < g.as[l + 1].rows; m++)
            {
                for (size_t b = 0; b < g.as[l + 1].cols; b++)
                {
                    MAT_AT(g.as[l + 1], m, b) = MAT_AT(g.as[l + 1], m, b) * MAT_AT(nn.as[l + 1], m, b) * (1 - MAT_AT(nn.as[l + 1], m, b));
                }
            }

            mat_sum(g.bs[l], g.as[l + 1]);

            Mat a_t = mat_t(nn.as[l]);
            Mat dCdw = mat_alloc(a_t.rows, g.as[l + 1].cols);
            mat_dot(dCdw, a_t, g.as[l + 1]);
            mat_sum(g.ws[l], dCdw);

            Mat w_t = mat_t(nn.ws[l]);
            Mat dCda = mat_alloc(g.as[l + 1].rows, w_t.cols);
            mat_dot(dCda, g.as[l + 1], w_t);
            mat_sum(g.as[l], dCda);

            free(a_t.es);
            free(dCdw.es);
            free(w_t.es);
            free(dCda.es);
        }
    }

    for (size_t i = 0; i < g.count; i++)
    {
        for (size_t j = 0; j < g.ws[i].rows; j++)
        {
            for (size_t k = 0; k < g.ws[i].cols; k++)
            {
                MAT_AT(g.ws[i], j, k) = MAT_AT(g.ws[i], j, k) / n;
            }
        }

        for (size_t j = 0; j < g.bs[i].rows; j++)
        {
            for (size_t k = 0; k < g.bs[i].cols; k++)
            {
                MAT_AT(g.bs[i], j, k) = MAT_AT(g.bs[i], j, k) / n;
            }
        }
    }
}

void nn_learn(NN nn, NN g, float rate)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        for (size_t j = 0; j < nn.ws[i].rows; j++)
        {
            for (size_t k = 0; k < nn.ws[i].cols; k++)
            {
                MAT_AT(nn.ws[i], j, k) -= rate * MAT_AT(g.ws[i], j, k);
            }
        }

        for (size_t j = 0; j < nn.bs[i].rows; j++)
        {
            for (size_t k = 0; k < nn.bs[i].cols; k++)
            {
                MAT_AT(nn.bs[i], j, k) -= rate * MAT_AT(g.bs[i], j, k);
            }
        }
    }
}

void nn_zero(NN nn)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        mat_fill(nn.ws[i], 0);
        mat_fill(nn.bs[i], 0);
    }
    for (size_t i = 0; i < nn.count + 1; i++)
    {
        mat_fill(nn.as[i], 0);
    }
}
#endif // SYNAPSE_IMPLEMENTATION
