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

#define IMG_WIDTH 800
#define IMG_HEIGHT 600

typedef int Errno;

static Errno file_size(FILE *file, size_t *size)
{
    long saved = ftell(file);
    if (saved < 0)
        return errno;
    if (fseek(file, 0, SEEK_END) < 0)
        return errno;
    long result = ftell(file);
    if (result < 0)
        return errno;
    if (fseek(file, saved, SEEK_SET) < 0)
        return errno;
    *size = (size_t)result;
    return 0;
}

// Errno read_entire_file(const char *file_path, String_Builder *sb)
// {
//     Errno result = 0;
//     FILE *f = NULL;

//     f = fopen(file_path, "r");
//     if (f == NULL)
//         return_defer(errno);

//     size_t size;
//     Errno err = file_size(f, &size);
//     if (err != 0)
//         return_defer(err);

//     if (sb->capacity < size)
//     {
//         sb->capacity = size;
//         sb->items = realloc(sb->items, sb->capacity * sizeof(*sb->items));
//         assert(sb->items != NULL && "Buy more RAM lol");
//     }

//     fread(sb->items, size, 1, f);
//     if (ferror(f))
//         return_defer(errno);
//     sb->count = size;

// defer:
//     if (f)
//         fclose(f);
//     return result;
// }

typedef struct
{
    size_t count;
    size_t capacity;
    size_t *items;
} Arch;

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

void nn_render_raylib(NN nn)
{
    Color background_color = {0x18, 0x18, 0x18, 0xFF};
    Color low = BLUE;
    Color high = RED;

    ClearBackground(background_color);

    int arch_count = nn.count + 1;

    for (size_t l = 0; l < arch_count; l++)
    {
        int layer_count = nn.as[l].cols;
        float neuron_radius = 20.f;
        int layer_border_pad = 20;

        int layer_height = IMG_HEIGHT - 2 * layer_border_pad;
        int nn_width = IMG_WIDTH - 2 * layer_border_pad;

        int layer_vpad = layer_height / (layer_count + 1);
        int nn_hpad = nn_width / (arch_count + 1);

        int nn_x = IMG_WIDTH / 2 - nn_width / 2;
        int nn_y = IMG_HEIGHT / 2 - layer_height / 2;

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
                    high.a = floorf(sigmoidf(MAT_AT(nn.ws[l], i, j)) * 255.f);
                    connection_color = ColorAlphaBlend(low, high, WHITE);
                    DrawLine(cx1, cy1, cx2, cy2, connection_color);
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

    const char *programm = args_shift(&argc, &argv);
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

    size_t stride = 3;

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

    float rate = 1e-1;

    NN nn = nn_alloc(arch.items, arch.count);
    NN g = nn_alloc(arch.items, arch.count);

    nn_rand(nn, -2, 2);

    InitWindow(IMG_WIDTH, IMG_HEIGHT, "gym");

    SetTargetFPS(60);

    size_t i = 0;
    while (!WindowShouldClose())
    {
        if (i < 5000)
        {
            for (size_t j = 0; j < 10; j++)
            {
                nn_backprop(nn, g, ti, to);
                nn_learn(nn, g, rate);
            }
            i++;

            char buf[256];
            snprintf(buf, sizeof(buf), "%zu: cost = %f\n", i, nn_cost(nn, ti, to));
            DrawText(buf, 29, 25, 20, WHITE);
        }

        BeginDrawing();
        nn_render_raylib(nn);
        EndDrawing();
    }
    CloseWindow();

    return 0;
}