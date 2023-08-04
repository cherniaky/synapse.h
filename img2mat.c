#include <stdio.h>
#include <assert.h>
#include <raylib.h>

#include "dev_deps/stb_image.h"
#include "dev_deps/stb_image_write.h"

#define SYNAPSE_IMPLEMENTATION
#include "synapse.h"

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
    float y_start = y_offset + render_h;
    float x_start = x_offset;
    float y_end = y_offset + render_h;
    float x_end = x_offset + render_w;
    DrawLine((int)x_start, (int)y_start, (int)x_end, (int)y_end, WHITE);
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
        fprintf(stderr, "ERROR: no input file was provided\n");
        return 1;
    }

    const char *img_file_path = args_shift(&argc, &argv);

    int img_width, img_height, n;
    uint8_t *img_pixels = (uint8_t *)stbi_load(img_file_path, &img_width, &img_height, &n, 0);
    if (img_pixels == NULL)
    {
        fprintf(stderr, "Could not read image %s\n", img_file_path);
        return 1;
    }
    if (n != 1)
    {
        fprintf(stderr, "Image %s is %d bits image. Only 8 bit grayscale images is supported\n", img_file_path, 8 * n);
        return 1;
    }
    fprintf(stdout, "%s is %dx%d %d bits\n", img_file_path, img_width, img_height, n * 8);

    Mat td = mat_alloc(img_width * img_height, 3);

    for (size_t y = 0; y < img_height; y++)
    {
        for (size_t x = 0; x < img_width; x++)
        {
            float nx = (float)x / (img_width - 1);
            float ny = (float)y / (img_height - 1);
            size_t i = y * img_width + x;
            float nv = img_pixels[i] / 255.f;

            MAT_AT(td, i, 0) = nx;
            MAT_AT(td, i, 1) = ny;
            MAT_AT(td, i, 2) = nv;
        }
    }

    size_t arch[] = {2, 7, 8, 4, 1};

    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, -2, 2);

    int WINDOW_FACTOR = 80;
    int WINDOW_HEIGHT = 9 * WINDOW_FACTOR;
    int WINDOW_WIDTH = 16 * WINDOW_FACTOR;
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "gym");

    SetTargetFPS(60);

    img_height = 64;
    img_width = 64;
    Image preview_image = GenImageColor(img_width, img_height, BLACK);
    Texture2D preview_texture = LoadTextureFromImage(preview_image);

    float rate = 2;
    size_t batch_size = 50;
    size_t batch_count = (td.rows + batch_size - 1) / batch_size;
    size_t epochs = 0;
    Cost_Plot cost_da = {0};
    size_t max_epoch = 100000;
    size_t epoch_per_frame = 10;
    bool paused = false;

    while (!WindowShouldClose())
    {
        if (IsKeyPressed(KEY_SPACE))
        {
            paused = !paused;
        }

        WINDOW_HEIGHT = GetRenderHeight();
        WINDOW_WIDTH = GetRenderWidth();
        float epoch_cost = 0;
        size_t training_size = batch_size;
        if (batch_size > td.rows)
        {
            training_size = td.rows;
        }

        if (epochs < max_epoch)
        {
            for (size_t j = 0; j < 5 * batch_count && !paused; j++)
            {
                mat_shuffle_rows(td);

                Mat batch_ti = {
                    .rows = training_size,
                    .cols = 2,
                    .stride = td.stride,
                    .es = &MAT_AT(td, 0, 0),
                };

                Mat batch_to = {
                    .rows = training_size,
                    .cols = 1,
                    .stride = td.stride,
                    .es = &MAT_AT(td, 0, batch_ti.cols),
                };

                nn_backprop(nn, g, batch_ti, batch_to);
                nn_learn(nn, g, rate);

                epoch_cost += nn_cost(nn, batch_ti, batch_to) / batch_count;
            }
            epochs++;
        }

        BeginDrawing();
        Color background_color = {0x18, 0x18, 0x18, 0xFF};
        ClearBackground(background_color);

        char buf[256];
        snprintf(buf, sizeof(buf), "%zu: cost = %f\n", epochs, epoch_cost);
        if (epochs % 100 == 0)
        {
            da_append(&cost_da, epoch_cost);
        }

        DrawText(buf, 29, 25, 20, WHITE);

        int render_w, render_h, x_offset, y_offset;

        render_h = WINDOW_HEIGHT / 2;
        render_w = WINDOW_WIDTH / 3;
        y_offset = (WINDOW_HEIGHT - render_h) / 2;
        x_offset = 0;
        if (cost_da.count > 0)
        {
            plot_cost(cost_da, x_offset, y_offset, render_w, render_h);
        }

        x_offset += render_w;
        nn_render_raylib(nn, x_offset, y_offset, render_w, render_h);

        x_offset += render_w;
        {
            for (size_t y = 0; y < img_height; y++)
            {
                for (size_t x = 0; x < img_width; x++)
                {
                    float nx = (float)x / (img_width - 1);
                    float ny = (float)y / (img_height - 1);

                    MAT_AT(NN_INPUT(nn), 0, 0) = nx;
                    MAT_AT(NN_INPUT(nn), 0, 1) = ny;
                    nn_forward(nn);
                    uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0) * 255.f;
                    ImageDrawPixel(&preview_image, x, y, CLITERAL(Color){pixel, pixel, pixel, 255});
                    // preview_texture = LoadTextureFromImage(preview_image);
                }
                UpdateTexture(preview_texture, preview_image.data);
                DrawTextureEx(preview_texture, CLITERAL(Vector2){x_offset, y_offset}, 0.f, 5, WHITE);
            }
        }
        EndDrawing();
    }
    CloseWindow();

    // for (size_t y = 0; y < img_height; y++)
    // {
    //     for (size_t x = 0; x < img_width; x++)
    //     {
    //         float nx = (float)x / (img_width - 1);
    //         float ny = (float)y / (img_height - 1);

    //         MAT_AT(NN_INPUT(nn), 0, 0) = nx;
    //         MAT_AT(NN_INPUT(nn), 0, 1) = ny;
    //         nn_forward(nn);
    //         uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0) * 255.f;

    //         if (pixel)
    //             printf("%3u ", pixel);
    //         else
    //             printf("   ");
    //     }
    //     printf("\n");
    // }

    size_t out_width = 512;
    size_t out_height = 512;
    uint8_t *out_pixels = malloc(sizeof(*out_pixels) * out_width * out_height);
    S_ASSERT(out_pixels != NULL);

    for (size_t y = 0; y < out_height; y++)
    {
        for (size_t x = 0; x < out_width; x++)
        {
            float nx = (float)x / (out_width - 1);
            float ny = (float)y / (out_height - 1);

            MAT_AT(NN_INPUT(nn), 0, 0) = nx;
            MAT_AT(NN_INPUT(nn), 0, 1) = ny;
            nn_forward(nn);
            uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0) * 255.f;

            out_pixels[y * out_width + x] = pixel;
        }
    }

    char *out_file_path = "out.png";
    if (!stbi_write_png(out_file_path, out_width, out_height, 1, out_pixels, out_width * sizeof(*out_pixels)))
    {
        fprintf(stderr, "ERROR: could not save image %s \n", out_file_path);
        return 1;
    }

    printf("Generated %s\n ", out_file_path);
    return 0;
}
