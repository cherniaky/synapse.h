#ifndef SYNAPSE_H_
#define SYNAPSE_H_

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifndef NN_ACT
#define NN_ACT ACT_SIG
#endif // NN_ACT

#ifndef NN_RELU_PARAM
#define NN_RELU_PARAM 0.01f
#endif // NN_RELU_PARAM

#ifndef S_CALLOC
#define S_CALLOC calloc
#endif // S_CALLOC

#ifndef S_ASSERT
#include <assert.h>
#define S_ASSERT assert
#endif // S_ASSERT

#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])

typedef enum
{
    ACT_SIG,
    ACT_RELU,
    ACT_TANH,
    ACT_SIN
} Act;

float rand_float(void);
float sigmoidf(float x);
float reluf(float x);
float tanhf(float x);

typedef struct
{
    size_t capacity;
    size_t size;
    char *data;
} Region;

Region region_alloc_alloc(size_t capacity);
void *region_alloc(Region *r, size_t size);
#define region_reset(r) (r)->size = 0

typedef struct
{
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i) * (m).stride + (j)]

Mat mat_alloc(Region *r, size_t rows, size_t cols);
void mat_save(FILE *out, Mat m);
Mat mat_load(FILE *in, Region *r);
void mat_dot(Mat dist, Mat a, Mat b);
void mat_sum(Mat dist, Mat a);
void mat_act(Mat m);
void mat_print(Mat m, const char *name, size_t padding);
void mat_rand(Mat m, float low, float high);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dist, Mat src);
void mat_fill(Mat m, float x);
#define MAT_PRINT(m) mat_print(m, #m, 0)
Mat mat_t(Region *r, Mat old);
void mat_shuffle_rows(Mat m);

typedef struct
{
    size_t *arch;
    size_t arch_count;
    Mat *ws, *bs, *as;
} NN;

#define NN_INPUT(nn) (S_ASSERT((nn).arch_count > 0), (nn).as[0])
#define NN_OUTPUT(nn) (S_ASSERT((nn).arch_count > 0), (nn).as[(nn).arch_count - 1])

NN nn_alloc(Region *r, size_t *arch, size_t arch_count);
void nn_print(NN nn, char *name);
#define NN_PRINT(nn) nn_print((nn), #nn)
void nn_rand(NN nn, float low, float high);
void nn_forward(NN nn);
float nn_cost(NN nn, Mat ti, Mat to);
void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to);
NN nn_backprop(Region *r, NN nn, Mat ti, Mat to);
void nn_learn(NN nn, NN g, float rate);
void nn_zero(NN nn);

typedef struct
{
    size_t begin;
    float cost;
    bool finished;
} Batch;

void batch_process(Region *r, Batch *b, size_t batch_size, NN nn, Mat t, float rate);

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

float reluf(float x)
{
    return x > 0 ? x : x * NN_RELU_PARAM;
}

float tanhf(float x)
{
    float ex = expf(x);
    float enx = expf(-x);
    return (ex - enx) / (ex + enx);
}

Mat mat_alloc(Region *r, size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = region_alloc(r, rows * cols * sizeof(*m.es));
    S_ASSERT(m.es != NULL);
    return m;
}

void mat_save(FILE *out, Mat m)
{
    const char *magic = "nn.h.mat";
    fwrite(magic, strlen(magic), 1, out);

    fwrite(&m.rows, sizeof(m.rows), 1, out);
    fwrite(&m.cols, sizeof(m.cols), 1, out);
    fwrite(&m.stride, sizeof(m.stride), 1, out);
    size_t n = fwrite(m.es, sizeof(*m.es), m.rows * m.cols, out);
    while (n < m.rows * m.cols && !ferror(out))
    {
        size_t j = fwrite(m.es + n, sizeof(*m.es), m.rows * m.cols - n, out);
        n += j;
    }
}

Mat mat_load(FILE *in, Region *r)
{
    uint64_t magic;
    fread(&magic, sizeof(magic), 1, in);
    S_ASSERT(magic == 0x74616d2e682e6e6e);
    size_t rows, cols, stride;
    fread(&rows, sizeof(rows), 1, in);
    fread(&cols, sizeof(cols), 1, in);
    fread(&stride, sizeof(stride), 1, in);
    Mat m = mat_alloc(r, rows, cols);
    m.stride = stride;

    size_t n = fread(m.es, sizeof(*m.es), m.rows * m.cols, in);
    while (n < rows * cols && !ferror(in))
    {
        size_t k = fread(m.es + n, sizeof(*m.es), m.rows * m.cols - n, in);
        n += k;
    }

    return m;
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

void mat_act(Mat m)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            switch (NN_ACT)
            {
            case ACT_SIG:
                MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
                break;
            case ACT_RELU:
                MAT_AT(m, i, j) = reluf(MAT_AT(m, i, j));
                break;
            case ACT_TANH:
                MAT_AT(m, i, j) = tanhf(MAT_AT(m, i, j));
                break;
            case ACT_SIN:
                MAT_AT(m, i, j) = sinf(MAT_AT(m, i, j));
                break;
            default:
                S_ASSERT(0 && "unreacheble");
            }
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
Mat mat_t(Region *r, Mat old)
{
    Mat mat = mat_alloc(r, old.cols, old.rows);
    for (size_t i = 0; i < old.rows; i++)
    {
        for (size_t j = 0; j < old.cols; j++)
        {
            MAT_AT(mat, j, i) = MAT_AT(old, i, j);
        }
    }
    return mat;
}

void mat_shuffle_rows(Mat m)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        size_t j = i + rand() % (m.rows - i);

        if (i != j)
        {
            for (size_t k = 0; k < m.cols; k++)
            {
                float t = MAT_AT(m, j, k);
                MAT_AT(m, j, k) = MAT_AT(m, i, k);
                MAT_AT(m, i, k) = t;
            }
        }
    }
}

NN nn_alloc(Region *r, size_t *arch, size_t arch_count)
{
    S_ASSERT(arch_count > 0);

    NN nn;
    nn.arch = arch;
    nn.arch_count = arch_count;

    nn.ws = region_alloc(r, (nn.arch_count - 1) * sizeof(*nn.ws));
    S_ASSERT(nn.ws != NULL);
    nn.bs = region_alloc(r, (nn.arch_count - 1) * sizeof(*nn.bs));
    S_ASSERT(nn.bs != NULL);
    nn.as = region_alloc(r, nn.arch_count * sizeof(*nn.as));
    S_ASSERT(nn.as != NULL);

    nn.as[0] = mat_alloc(r, 1, arch[0]);
    for (size_t i = 1; i < arch_count; i++)
    {
        nn.ws[i - 1] = mat_alloc(r, nn.as[i - 1].cols, arch[i]);
        nn.bs[i - 1] = mat_alloc(r, nn.as[i - 1].rows, arch[i]);
        nn.as[i] = mat_alloc(r, nn.as[i - 1].rows, arch[i]);
    }

    return nn;
}

void nn_print(NN nn, char *name)
{
    char buf[256];

    printf("%s = [\n", name);
    Mat *ws = nn.ws;
    Mat *bs = nn.bs;
    for (size_t i = 0; i < nn.arch_count - 1; i++)
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
    for (size_t i = 0; i < nn.arch_count - 1; i++)
    {
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

void nn_forward(NN nn)
{
    for (size_t i = 0; i < nn.arch_count - 1; i++)
    {
        mat_dot(nn.as[i + 1], nn.as[i], nn.ws[i]);
        mat_sum(nn.as[i + 1], nn.bs[i]);
        mat_act(nn.as[i + 1]);
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

    for (size_t i = 0; i < nn.arch_count - 1; i++)
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

NN nn_backprop(Region *r, NN nn, Mat ti, Mat to)
{
    S_ASSERT(ti.rows == to.rows);
    size_t n = ti.rows;
    S_ASSERT(NN_OUTPUT(nn).cols == to.cols);

    NN g = nn_alloc(r, nn.arch, nn.arch_count);
    nn_zero(g);
    // i - current sample
    for (size_t i = 0; i < n; i++)
    {
        mat_copy(NN_INPUT(nn), mat_row(ti, i));
        nn_forward(nn);

        for (size_t j = 0; j < nn.arch_count; j++)
        {
            mat_fill(g.as[j], 0);
        }

        for (size_t j = 0; j < to.cols; j++)
        {
            MAT_AT(NN_OUTPUT(g), 0, j) = 2 * (MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(to, i, j)) / n;
        }

        for (int l = nn.arch_count - 2; l >= 0; l--)
        {
            for (size_t m = 0; m < g.as[l + 1].rows; m++)
            {
                for (size_t b = 0; b < g.as[l + 1].cols; b++)
                {
                    float v = MAT_AT(nn.as[l + 1], m, b);
                    float q = 0;
                    switch (NN_ACT)
                    {
                    case ACT_SIG:
                        q = v * (1 - v);
                        break;
                    case ACT_RELU:
                        q = v >= 0 ? 1 : NN_RELU_PARAM;
                        break;
                    case ACT_TANH:
                        q = 1 - v * v;
                        break;
                    case ACT_SIN:
                        q = sqrtf(1 - v * v);
                        break;
                    default:
                        S_ASSERT(0 && "unreacheble");
                    }
                    MAT_AT(g.as[l + 1], m, b) = q * MAT_AT(g.as[l + 1], m, b);
                }
            }

            mat_sum(g.bs[l], g.as[l + 1]);

            Mat a_t = mat_t(r, nn.as[l]);
            Mat dCdw = mat_alloc(r, a_t.rows, g.as[l + 1].cols);
            mat_dot(dCdw, a_t, g.as[l + 1]);
            mat_sum(g.ws[l], dCdw);

            Mat w_t = mat_t(r, nn.ws[l]);
            Mat dCda = mat_alloc(r, g.as[l + 1].rows, w_t.cols);
            mat_dot(dCda, g.as[l + 1], w_t);
            mat_sum(g.as[l], dCda);

        }
    }

    return g;
}

void nn_learn(NN nn, NN g, float rate)
{
    for (size_t i = 0; i < nn.arch_count - 1; i++)
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
    for (size_t i = 0; i < nn.arch_count - 1; i++)
    {
        mat_fill(nn.ws[i], 0);
        mat_fill(nn.bs[i], 0);
        mat_fill(nn.as[i], 0);
    }
    mat_fill(nn.as[nn.arch_count - 1], 0);
}

void batch_process(Region *r, Batch *b, size_t batch_size, NN nn, Mat t, float rate)
{
    if (b->finished)
    {
        b->finished = false;
        b->begin = 0;
        b->cost = 0;
    }

    size_t size = batch_size;
    if (b->begin + batch_size >= t.rows)
    {
        size = t.rows - b->begin;
    }

    Mat batch_ti = {
        .rows = size,
        .cols = NN_INPUT(nn).cols,
        .stride = t.stride,
        .es = &MAT_AT(t, b->begin, 0),
    };

    Mat batch_to = {
        .rows = size,
        .cols = NN_OUTPUT(nn).cols,
        .stride = t.stride,
        .es = &MAT_AT(t, b->begin, batch_ti.cols),
    };

    NN g = nn_backprop(r, nn, batch_ti, batch_to);
    nn_learn(nn, g, rate);
    b->cost += nn_cost(nn, batch_ti, batch_to);
    b->begin += batch_size;

    if (b->begin >= t.rows)
    {
        size_t batch_count = (t.rows + batch_size - 1) / batch_size;
        b->cost /= batch_count;
        b->finished = true;
    }
}

Region region_alloc_alloc(size_t capacity)
{
    void *data = S_CALLOC(capacity, 1);
    S_ASSERT(data != NULL);
    Region r = {
        .capacity = capacity,
        .data = data,
        .size = 0,
    };
    return r;
}

void *region_alloc(Region *r, size_t size)
{
    if (r == NULL)
        return S_CALLOC(size, 1);

    S_ASSERT(r->size + size <= r->capacity);
    if (r->size + size > r->capacity) return NULL;

    void *result = &r->data[r->size];

    r->size += size;

    return result;
}

#endif // SYNAPSE_IMPLEMENTATION
