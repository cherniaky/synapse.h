#include <time.h>
#include <stdint.h>

#define SYNAPSE_IMPLEMENTATION ;
#include "synapse.h"
#include "raylib.h"

float td[] = {
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    0,
    1,
    1,
    1,
    0,
};

#define IMG_FACTOR 80
#define IMG_WIDTH (16 * IMG_FACTOR)
#define IMG_HEIGHT (9 * IMG_FACTOR)

uint32_t img_pixels[IMG_HEIGHT * IMG_WIDTH];

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

int main()
{

    srand(time(0));

    float rate = 1e-1;

    size_t arch[] = {2, 4, 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));

    nn_rand(nn, -2, 2);

    size_t stride = 3;
    size_t n = sizeof(td) / sizeof(td[0]) / stride;
    Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td};

    Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = td + 2};

    InitWindow(IMG_WIDTH, IMG_HEIGHT, "xor.c");
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