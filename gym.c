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

int main(int argc, char **argv)
{
    unsigned int buffer_len = 0;
    unsigned char *buffer = LoadFileData("xor.arch", &buffer_len);

    String_View content = sv_from_parts((const char *)buffer, buffer_len);

    Arch arch = {0};

    content = sv_trim(content);
    while (content.count > 0 && isdigit(content.data[0]))
    {
        int x = sv_chop_u64(&content);
        printf("%d\n", x);
        da_append(&arch, x);
        content = sv_trim(content);
    }

    srand(time(0));

    float rate = 1e-1;

    // size_t arch[] = {2, 4, 1};
    NN nn = nn_alloc(arch.items, arch.count);
    NN g = nn_alloc(arch, ARRAY_LEN(arch));

    nn_rand(nn, -2, 2);
    nn_print(nn, "nn");
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

            // char buf[256];
            // snprintf(buf, sizeof(buf), "%zu: cost = %f\n", i, nn_cost(nn, ti, to));
            // DrawText(buf, 29, 25, 20, WHITE);
        }

        BeginDrawing();
        nn_render_raylib(nn);
        EndDrawing();
    }
    CloseWindow();

    return 0;
}