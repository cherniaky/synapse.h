// Gym is GUI app that trains your NN on the data you give it.
// The idea that it produce .exe that you can use

#include <raylib.h>
#include <stdio.h>
#include <stddef.h>
#include <errno.h>
#include <time.h>
#define SYNAPSE_IMPLEMENTATION
#include "synapse.h"
#define SV_IMPLEMENTATION
#include "dev_deps/sv.h"

// #define IMG_WIDTH 800
// #define IMG_HEIGHT 600

typedef int Errno;

// static Errno file_size(FILE *file, size_t *size)
// {
//     long saved = ftell(file);
//     if (saved < 0)
//         return errno;
//     if (fseek(file, 0, SEEK_END) < 0)
//         return errno;
//     long result = ftell(file);
//     if (result < 0)
//         return errno;
//     if (fseek(file, saved, SEEK_SET) < 0)
//         return errno;
//     *size = (size_t)result;
//     return 0;
// }

typedef struct
{
    size_t count;
    size_t capacity;
    size_t *items;
} Arch;

typedef struct
{
    size_t count;
    size_t capacity;
    float *items;
} Cost_Plot;

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

void nn_render_raylib(NN nn, int x_offset, int y_offset, int w, int h)
{
    Color low = BLUE;
    Color high = RED;

    int arch_count = nn.count + 1;

    for (size_t l = 0; l < arch_count; l++)
    {
        int layer_count = nn.as[l].cols;
        float neuron_radius = 0.04f * (float)h;
        int layer_border_pad = 20;

        int layer_height = h - 2 * layer_border_pad;
        int nn_width = w - 2 * layer_border_pad;

        int layer_vpad = layer_height / (layer_count + 1);
        int nn_hpad = nn_width / (arch_count + 1);

        int nn_x = w / 2 - nn_width / 2 + x_offset;
        int nn_y = h / 2 - layer_height / 2 + y_offset;

        for (size_t i = 0; i < layer_count; i++)
        {
            int cx1 = nn_x + (l + 1) * nn_hpad;
            int cy1 = nn_y + (i + 1) * layer_vpad;

            if (l < arch_count - 1)
            {
                for (size_t j = 0; j < nn.as[l + 1].cols; j++)
                {
                    int cx2 = nn_x + (l + 1 + 1) * nn_hpad;
                    int next_layer_vpad = layer_height / (nn.as[l + 1].cols + 1);
                    int cy2 = nn_y + (j + 1) * next_layer_vpad;

                    Color connection_color = low;
                    float w_activation = sigmoidf(MAT_AT(nn.ws[l], i, j));
                    high.a = floorf(w_activation * 255.f);
                    connection_color = ColorAlphaBlend(low, high, WHITE);
                    Vector2 start = {
                        .x = cx1,
                        .y = cy1,
                    };
                    Vector2 end = {
                        .x = cx2,
                        .y = cy2,
                    };
                    float thickness = h * 0.004f;
                    DrawLineEx(start, end, thickness * fabs(w_activation), connection_color);
                }
            }
            Color neuron_color = low;
            if (l == 0)
            {
                neuron_color = GRAY;
            }
            else
            {
                high.a = floorf(sigmoidf(MAT_AT(nn.bs[l - 1], 0, i)) * 255.f);
                neuron_color = ColorAlphaBlend(low, high, WHITE);
            }
            DrawCircle(cx1, cy1, neuron_radius, neuron_color);
        }
    }
}

float cost_plot_max(Cost_Plot plot)
{
    float max = 0;
    for (size_t i = 0; i < plot.count; i++)
    {
        if (max < plot.items[i])
        {
            max = plot.items[i];
        }
    }

    return max;
}

void plot_cost(Cost_Plot cost_da, int x_offset, int y_offset, int render_w, int render_h)
{
    float max = cost_plot_max(cost_da) * 1.f;
    size_t n = cost_da.count;
    if (n < 100)
        n = 100;
    float x_padding = (float)(render_w / n);

    for (size_t i = 1; i < cost_da.count; i++)
    {
        float y_start = y_offset + (1 - cost_da.items[i - 1] / max) * render_h;
        float x_start = x_offset + x_padding * (i - 1);
        float y_end = y_offset + (1 - cost_da.items[i] / max) * render_h;
        float x_end = x_offset + x_padding * (i);
        DrawLine((int)x_start, (int)y_start, (int)x_end, (int)y_end, WHITE);
    }
}

char *args_shift(int *argc, char ***argv)
{
    assert(*argc > 0);
    char *result = **argv;
    (*argc)--;
    (*argv)++;
    return result;
}

int main(int argc, char **argv)
{
    srand(time(0));

    args_shift(&argc, &argv);
    if (argc < 0)
    {
        fprintf(stderr, "ERROR: no arch file was provided\n");
        return 1;
    }

    const char *arch_file_path = args_shift(&argc, &argv);
    if (argc < 0)
    {
        fprintf(stderr, "ERROR: no data file was provided\n");
        return 1;
    }

    const char *data_file_path = args_shift(&argc, &argv);

    unsigned int buffer_len = 0;
    unsigned char *buffer = LoadFileData(arch_file_path, &buffer_len);
    if (buffer == NULL)
    {
        return 1;
    }

    String_View content = sv_from_parts((const char *)buffer, buffer_len);

    Arch arch = {0};

    content = sv_trim(content);
    while (content.count > 0 && isdigit(content.data[0]))
    {
        int x = sv_chop_u64(&content);
        da_append(&arch, x);
        content = sv_trim(content);
    }

    FILE *in = fopen(data_file_path, "rb");
    if (ferror(in))
    {
        fprintf(stderr, "ERROR:Could not read file %s \n", data_file_path);
        return 1;
    }

    Mat td = mat_load(in);
    fclose(in);

    size_t output_count = arch.items[arch.count - 1];

    S_ASSERT(td.cols == arch.items[0] + output_count);

    Mat ti = {
        .rows = td.rows,
        .cols = arch.items[0],
        .stride = td.stride,
        .es = td.es};

    Mat to = {
        .rows = td.rows,
        .cols = output_count,
        .stride = td.stride,
        .es = td.es + (td.cols - output_count)};

    float rate = 1e-4;

    NN nn = nn_alloc(arch.items, arch.count);
    NN g = nn_alloc(arch.items, arch.count);

    nn_rand(nn, -2, 2);
    int IMG_HEIGHT = 1000;
    int IMG_WIDTH = 1500;
    InitWindow(IMG_WIDTH, IMG_HEIGHT, "gym");

    SetTargetFPS(60);

    size_t i = 0;
    Cost_Plot cost_da = {0};

    while (!WindowShouldClose())
    {
        IMG_HEIGHT = GetRenderHeight();
        IMG_WIDTH = GetRenderWidth();
        if (i < 5000)
        {
            for (size_t j = 0; j < 10; j++)
            {
                nn_backprop(nn, g, ti, to);
                nn_learn(nn, g, rate);
            }
            i++;

            char buf[256];
            float i_cost = nn_cost(nn, ti, to);
            snprintf(buf, sizeof(buf), "%zu: cost = %f\n", i, i_cost);
            if (i % 10 == 0)
            {
                da_append(&cost_da, i_cost);
            }

            DrawText(buf, 29, 25, 20, WHITE);
        }

        BeginDrawing();
        Color background_color = {0x18, 0x18, 0x18, 0xFF};
        ClearBackground(background_color);
        int render_w, render_h, x_offset, y_offset;

        render_w = IMG_WIDTH * 0.6;
        render_h = IMG_HEIGHT;
        x_offset = IMG_WIDTH - render_w;
        y_offset = (IMG_HEIGHT - render_h) / 2;
        nn_render_raylib(nn, x_offset, y_offset, render_w, render_h);

        render_w = IMG_WIDTH * 0.4;
        render_h = IMG_HEIGHT / 2;
        x_offset = 30;
        y_offset = (IMG_HEIGHT - render_h) / 2;
        if (cost_da.count > 0)
        {
            plot_cost(cost_da, x_offset, y_offset, render_w, render_h);
        }
        EndDrawing();
    }
    CloseWindow();

    return 0;
}