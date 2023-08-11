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
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i) * (m).stride + (j)]

Mat mat_alloc(size_t rows, size_t cols);
void mat_save(FILE *out, Mat m);
Mat mat_load(FILE *in);
void mat_dot(Mat dist, Mat a, Mat b);
void mat_sum(Mat dist, Mat a);
void mat_act(Mat m);
void mat_print(Mat m, const char *name, size_t padding);
void mat_rand(Mat m, float low, float high);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dist, Mat src);
void mat_fill(Mat m, float x);
#define MAT_PRINT(m) mat_print(m, #m, 0)
Mat mat_t(Mat old);
void mat_shuffle_rows(Mat m);

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

typedef struct
{
    size_t begin;
    float cost;
    bool finished;
} Batch;

void batch_process(Batch *b, size_t batch_size, NN nn, NN g, Mat t, float rate);

#ifdef SYNAPSE_ENABLE_GYM
#include <float.h>
#include <raylib.h>
#include <raymath.h>

typedef struct
{
    float *items;
    size_t count;
    size_t capacity;
} Gym_Plot;

typedef struct
{
    float x;
    float y;
    float w;
    float h;
} Gym_Rect;

typedef enum
{
    GLO_HORZ,
    GLO_VERT,
} Gym_Layout_Orient;

typedef struct
{
    Gym_Layout_Orient orient;
    Gym_Rect rect;
    size_t count;
    size_t i;
    float gap;
} Gym_Layout;

typedef struct
{
    Gym_Layout *items;
    size_t count;
    size_t capacity;
} Gym_Layout_Stack;

Gym_Rect gym_rect(float x, float y, float w, float h);

Gym_Rect gym_layout_slot_loc(Gym_Layout *l, const char *filename, int line);

void gym_layout_stack_push(Gym_Layout_Stack *ls, Gym_Layout_Orient orient, Gym_Rect rect, size_t count, float gap);
#define gym_layout_stack_slot(ls) (assert((ls)->count > 0), gym_layout_slot_loc(&(ls)->items[(ls)->count - 1], __FILE__, __LINE__))
#define gym_layout_stack_pop(ls) \
    do                           \
    {                            \
        assert((ls)->count > 0); \
        (ls)->count -= 1;        \
    } while (0)

static Gym_Layout_Stack default_gym_layout_stack = {0};

#define gym_layout_begin(orient, rect, count, gap) gym_layout_stack_push(&default_gym_layout_stack, (orient), (rect), (count), (gap))
#define gym_layout_end() gym_layout_stack_pop(&default_gym_layout_stack)
#define gym_layout_slot() gym_layout_stack_slot(&default_gym_layout_stack)

#define DA_INIT_CAP 256
#define da_append(da, item)                                                            \
    do                                                                                 \
    {                                                                                  \
        if ((da)->count >= (da)->capacity)                                             \
        {                                                                              \
            (da)->capacity = (da)->capacity == 0 ? DA_INIT_CAP : (da)->capacity * 2;   \
            (da)->items = realloc((da)->items, (da)->capacity * sizeof(*(da)->items)); \
            assert((da)->items != NULL && "Buy more RAM lol");                         \
        }                                                                              \
                                                                                       \
        (da)->items[(da)->count++] = (item);                                           \
    } while (0)

void gym_render_nn(NN nn, float rx, float ry, float rw, float rh);
void gym_plot(Gym_Plot plot, int rx, int ry, int rw, int rh);
void gym_slider(float *value, bool *dragging, float rx, float ry, float rw, float rh);
void gym_nn_image_grayscale(NN nn, void *pixels, size_t width, size_t height, size_t stride, float low, float high);

#endif // SYNAPSE_ENABLE_GYM
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

Mat mat_load(FILE *in)
{
    uint64_t magic;
    fread(&magic, sizeof(magic), 1, in);
    S_ASSERT(magic == 0x74616d2e682e6e6e);
    size_t rows, cols, stride;
    fread(&rows, sizeof(rows), 1, in);
    fread(&cols, sizeof(cols), 1, in);
    fread(&stride, sizeof(stride), 1, in);
    Mat m = mat_alloc(rows, cols);
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
            MAT_AT(NN_OUTPUT(g), 0, j) = 2 * (MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(to, i, j)) / n;
        }

        for (int l = nn.count - 1; l >= 0; l--)
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
        mat_fill(nn.as[i], 0);
    }
    mat_fill(nn.as[nn.count], 0);
}

void batch_process(Batch *b, size_t batch_size, NN nn, NN g, Mat t, float rate)
{
    if (b->finished) {
        b->finished = false;
        b->begin = 0;
        b->cost = 0;
    }

    size_t size = batch_size;
    if (b->begin + batch_size >= t.rows)  {
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

    nn_backprop(nn, g, batch_ti, batch_to);
    nn_learn(nn, g, rate);
    b->cost += nn_cost(nn, batch_ti, batch_to);
    b->begin += batch_size;

    if (b->begin >= t.rows) {
        size_t batch_count = (t.rows + batch_size - 1)/batch_size;
        b->cost /= batch_count;
        b->finished = true;
    }
}

#ifdef SYNAPSE_ENABLE_GYM
void gym_render_nn(NN nn, float rx, float ry, float rw, float rh)
{
    Color low_color = {0xFF, 0x00, 0xFF, 0xFF};
    Color high_color = {0x00, 0xFF, 0x00, 0xFF};
    float neuron_radius = rh * 0.03;
    float layer_border_vpad = rh * 0.08;
    float layer_border_hpad = rw * 0.06;
    float nn_width = rw - 2 * layer_border_hpad;
    float nn_height = rh - 2 * layer_border_vpad;
    float nn_x = rx + rw / 2 - nn_width / 2;
    float nn_y = ry + rh / 2 - nn_height / 2;
    size_t arch_count = nn.count + 1;
    float layer_hpad = nn_width / arch_count;
    for (size_t l = 0; l < arch_count; ++l)
    {
        float layer_vpad1 = nn_height / nn.as[l].cols;
        for (size_t i = 0; i < nn.as[l].cols; ++i)
        {
            float cx1 = nn_x + l * layer_hpad + layer_hpad / 2;
            float cy1 = nn_y + i * layer_vpad1 + layer_vpad1 / 2;
            if (l + 1 < arch_count)
            {
                float layer_vpad2 = nn_height / nn.as[l + 1].cols;
                for (size_t j = 0; j < nn.as[l + 1].cols; ++j)
                {
                    // i - rows of ws
                    // j - cols of ws
                    float cx2 = nn_x + (l + 1) * layer_hpad + layer_hpad / 2;
                    float cy2 = nn_y + j * layer_vpad2 + layer_vpad2 / 2;
                    float value = sigmoidf(MAT_AT(nn.ws[l], i, j));
                    high_color.a = floorf(255.f * value);
                    float thick = rh * 0.004f;
                    Vector2 start = {cx1, cy1};
                    Vector2 end = {cx2, cy2};
                    DrawLineEx(start, end, thick, ColorAlphaBlend(low_color, high_color, WHITE));
                }
            }
            if (l > 0)
            {
                high_color.a = floorf(255.f * sigmoidf(MAT_AT(nn.bs[l - 1], 0, i)));
                DrawCircle(cx1, cy1, neuron_radius, ColorAlphaBlend(low_color, high_color, WHITE));
            }
            else
            {
                DrawCircle(cx1, cy1, neuron_radius, GRAY);
            }
        }
    }
}
void gym_plot(Gym_Plot plot, int rx, int ry, int rw, int rh)
{
    float min = FLT_MAX, max = FLT_MIN;
    for (size_t i = 0; i < plot.count; ++i)
    {
        if (max < plot.items[i])
            max = plot.items[i];
        if (min > plot.items[i])
            min = plot.items[i];
    }
    if (min > 0)
        min = 0;
    size_t n = plot.count;
    if (n < 1000)
        n = 1000;
    for (size_t i = 0; i + 1 < plot.count; ++i)
    {
        float x1 = rx + (float)rw / n * i;
        float y1 = ry + (1 - (plot.items[i] - min) / (max - min)) * rh;
        float x2 = rx + (float)rw / n * (i + 1);
        float y2 = ry + (1 - (plot.items[i + 1] - min) / (max - min)) * rh;
        DrawLineEx((Vector2){x1, y1}, (Vector2){x2, y2}, rh * 0.005, RED);
    }
    float y0 = ry + (1 - (0 - min) / (max - min)) * rh;
    DrawLineEx((Vector2){rx + 0, y0}, (Vector2){rx + rw - 1, y0}, rh * 0.005, WHITE);
    DrawText("0", rx + 0, y0 - rh * 0.04, rh * 0.04, WHITE);
}
void gym_slider(float *value, bool *dragging, float rx, float ry, float rw, float rh)
{
    float knob_radius = rh;
    Vector2 bar_size = {
        .x = rw - 2 * knob_radius,
        .y = rh * 0.25,
    };
    Vector2 bar_position = {
        .x = rx + knob_radius,
        .y = ry + rh / 2 - bar_size.y / 2};
    DrawRectangleV(bar_position, bar_size, WHITE);
    Vector2 knob_position = {
        .x = bar_position.x + bar_size.x * (*value),
        .y = ry + rh / 2};
    DrawCircleV(knob_position, knob_radius, RED);
    if (*dragging)
    {
        float x = GetMousePosition().x;
        if (x < bar_position.x)
            x = bar_position.x;
        if (x > bar_position.x + bar_size.x)
            x = bar_position.x + bar_size.x;
        *value = (x - bar_position.x) / bar_size.x;
    }
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
    {
        Vector2 mouse_position = GetMousePosition();
        if (Vector2Distance(mouse_position, knob_position) <= knob_radius)
        {
            *dragging = true;
        }
    }
    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT))
    {
        *dragging = false;
    }
}
void gym_nn_image_grayscale(NN nn, void *pixels, size_t width, size_t height, size_t stride, float low, float high)
{
    S_ASSERT(NN_INPUT(nn).cols >= 2);
    S_ASSERT(NN_OUTPUT(nn).cols >= 1);
    uint32_t *pixels_u32 = pixels;
    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            MAT_AT(NN_INPUT(nn), 0, 0) = (float)x / (float)(width - 1);
            MAT_AT(NN_INPUT(nn), 0, 1) = (float)y / (float)(height - 1);
            nn_forward(nn);
            float a = MAT_AT(NN_OUTPUT(nn), 0, 0);
            if (a < low)
                a = low;
            if (a > high)
                a = high;
            uint32_t pixel = (a - low) / (high - low) * 255.f;
            pixels_u32[y * stride + x] = (0xFF << (8 * 3)) | (pixel << (8 * 2)) | (pixel << (8 * 1)) | (pixel << (8 * 0));
        }
    }
}

Gym_Rect gym_rect(float x, float y, float w, float h)
{
    Gym_Rect r = {0};
    r.x = x;
    r.y = y;
    r.w = w;
    r.h = h;
    return r;
}

Gym_Rect gym_layout_slot_loc(Gym_Layout *l, const char *file_path, int line)
{
    if (l->i >= l->count)
    {
        fprintf(stderr, "%s:%d: ERROR: Layout overflow\n", file_path, line);
        exit(1);
    }

    Gym_Rect r = {0};

    switch (l->orient)
    {
    case GLO_HORZ:
        r.w = l->rect.w / l->count;
        r.h = l->rect.h;
        r.x = l->rect.x + l->i * r.w;
        r.y = l->rect.y;

        if (l->i == 0)
        { // First
            r.w -= l->gap / 2;
        }
        else if (l->i >= l->count - 1)
        { // Last
            r.x += l->gap / 2;
            r.w -= l->gap / 2;
        }
        else
        { // Middle
            r.x += l->gap / 2;
            r.w -= l->gap;
        }

        break;

    case GLO_VERT:
        r.w = l->rect.w;
        r.h = l->rect.h / l->count;
        r.x = l->rect.x;
        r.y = l->rect.y + l->i * r.h;

        if (l->i == 0)
        { // First
            r.h -= l->gap / 2;
        }
        else if (l->i >= l->count - 1)
        { // Last
            r.y += l->gap / 2;
            r.h -= l->gap / 2;
        }
        else
        { // Middle
            r.y += l->gap / 2;
            r.h -= l->gap;
        }

        break;

    default:
        assert(0 && "Unreachable");
    }

    l->i += 1;

    return r;
}

void gym_layout_stack_push(Gym_Layout_Stack *ls, Gym_Layout_Orient orient, Gym_Rect rect, size_t count, float gap)
{
    Gym_Layout l = {0};
    l.orient = orient;
    l.rect = rect;
    l.count = count;
    l.gap = gap;
    da_append(ls, l);
}

#endif // NN_ENABLE_GYM

#endif // SYNAPSE_IMPLEMENTATION
