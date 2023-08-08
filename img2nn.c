#include <stdio.h>
#include <assert.h>
#include <raylib.h>
#include <raymath.h>
#include <time.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

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

#define out_width 512
#define out_height 512
uint32_t *out_pixels = malloc(sizeof(*out_pixels) * out_width * out_height);

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

    size_t start = 1;

    if (cost_da.count > 200)
    {
        start = cost_da.count - 200;
        n = 200;
    }

    float x_padding = (float)(render_w / n);

    for (size_t i = start; i < cost_da.count; i++)
    {
        float y_start = y_offset + (1 - cost_da.items[i - 1] / max) * render_h;
        float x_start = x_offset + x_padding * (i - 1 - start);
        float y_end = y_offset + (1 - cost_da.items[i] / max) * render_h;
        float x_end = x_offset + x_padding * (i - start);
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

int render_upscale_video(NN nn, const char *out_file_path)
{
    int pipefd[2];

    if (pipe(pipefd) < 0)
    {
        fprintf(stderr, "ERROR: could not create a pipe: %s\n", strerror(errno));
        return 1;
    }

    pid_t child = fork();
    if (child < 0)
    {
        fprintf(stderr, "ERROR: could not fork a child: %s\n", strerror(errno));
        return 1;
    }

    if (child == 0)
    {
        if (dup2(pipefd[READ_END], STDIN_FILENO) < 0)
        {
            fprintf(stderr, "ERROR: could not reopen read end as stdin: %s\n", strerror(errno));
        }
        close(pipefd[WRITE_END]);

        int ret = execlp("ffmpeg", "ffmpeg",
                         "-loglevel", "verbose",
                         "-y",
                         "-f", "rawvideo",
                         "-pix_fmt", "rgba",
                         "-s", STR(WIDTH) "x" STR(HEIGHT),
                         "-r", STR(FPS),
                         "-an",
                         "-i", "-",
                         "-c:v", "libx264",
                         out_file_path,
                         NULL);
        if (ret < 0)
        {
            fprintf(stderr, "ERROR: could not run ffmpeg as child process: %s\n", strerror(errno));
        }

        assert(0 && "unreachable");
    }

    close(pipefd[READ_END]);

    // Olivec_Canvas oc = olivec_canvas(pixels, WIDTH, HEIGHT, WIDTH);

    // size_t duration = 10;
    // float x = WIDTH / 2;
    // float y = HEIGHT / 2;
    // float r = HEIGHT / 8;
    // float dx = 100;
    // float dy = 100;
    // float dt = 1.f / FPS;

    // for (size_t i = 0; i < FPS * duration; i++)
    // {
    //     float nx = x + dx * dt;
    //     if (0 + r < nx && nx < WIDTH - r)
    //     {
    //         x = nx;
    //     }
    //     else
    //     {
    //         dx = -dx;
    //     }

    //     float ny = y + dy * dt;
    //     if (0 + r < ny && ny < HEIGHT - r)
    //     {
    //         y = ny;
    //     }
    //     else
    //     {
    //         dy = -dy;
    //     }

    //     olivec_fill(oc, 0xFF181818);
    //     olivec_circle(oc, x, y, r, 0xFF0000FF);
    //     write(pipefd[WRITE_END], pixels, sizeof(*pixels) * WIDTH * HEIGHT);
    // }
    close(pipefd[WRITE_END]);

    wait(NULL);

    printf("Done rendering the video. The child's pid is %d\n", child);

    return 0;
}

void render_single_out_image(NN nn, float scroll)
{
    for (size_t y = 0; y < out_height; y++)
    {
        for (size_t x = 0; x < out_width; x++)
        {
            float nx = (float)x / (out_width - 1);
            float ny = (float)y / (out_height - 1);

            MAT_AT(NN_INPUT(nn), 0, 0) = nx;
            MAT_AT(NN_INPUT(nn), 0, 1) = ny;
            MAT_AT(NN_INPUT(nn), 0, 2) = scroll;
            nn_forward(nn);
            float activation = MAT_AT(NN_OUTPUT(nn), 0, 0);
            if (activation < 0)
            {
                activation = 0;
            }
            if (activation > 1)
            {
                activation = 1;
            }

            uint32_t bright = activation * 255.f;
            uint32_t pixel = 0xFF000000 | bright | (bright << 8) | (bright << 16);

            out_pixels[y * out_width + x] = pixel;
        }
    }
}

int render_upscale_screenshot(NN nn, char *out_file_path, float scroll)
{
    S_ASSERT(out_pixels != NULL);

    render_single_out_image(nn, scroll);

    if (!stbi_write_png(out_file_path, out_width, out_height, 1, out_pixels, out_width * sizeof(*out_pixels)))
    {
        fprintf(stderr, "ERROR: could not save image %s \n", out_file_path);
        return 1;
    }

    return 0;
}

int main(int argc, char **argv)
{
    srand(time(0));

    char *programm = args_shift(&argc, &argv);
    if (argc <= 0)
    {
        fprintf(stderr, "Usage: %s <img1> <img2>\n", programm);
        fprintf(stderr, "ERROR: no input file was provided\n");
        return 1;
    }

    const char *img1_file_path = args_shift(&argc, &argv);

    if (argc <= 0)
    {
        fprintf(stderr, "Usage: %s <img1> <img2>\n", programm);
        fprintf(stderr, "ERROR: no image2 was provided\n");
        return 1;
    }

    int img1_width, img1_height, n1;
    uint8_t *img1_pixels = (uint8_t *)stbi_load(img1_file_path, &img1_width, &img1_height, &n1, 0);
    if (img1_pixels == NULL)
    {
        fprintf(stderr, "Could not read image %s\n", img1_file_path);
        return 1;
    }
    if (n1 != 1)
    {
        fprintf(stderr, "Image %s is %d bits image. Only 8 bit grayscale images is supported\n", img1_file_path, 8 * n1);
        return 1;
    }
    fprintf(stdout, "%s is %dx%d %d bits\n", img1_file_path, img1_width, img1_height, n1 * 8);

    const char *img2_file_path = args_shift(&argc, &argv);

    int img2_width, img2_height, n2;
    uint8_t *img2_pixels = (uint8_t *)stbi_load(img2_file_path, &img2_width, &img2_height, &n2, 0);
    if (img2_pixels == NULL)
    {
        fprintf(stderr, "Could not read image %s\n", img2_file_path);
        return 1;
    }
    if (n2 != 1)
    {
        fprintf(stderr, "Image %s is %d bits image. Only 8 bit grayscale images is supported\n", img2_file_path, 8 * n2);
        return 1;
    }
    fprintf(stdout, "%s is %dx%d %d bits\n", img2_file_path, img2_width, img2_height, n2 * 8);

    size_t arch[] = {3, 7, 7, 6, 3, 1};

    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));

    Mat td = mat_alloc(img1_width * img1_height + img2_width * img2_height, NN_INPUT(nn).cols + NN_OUTPUT(nn).cols);

    for (size_t y = 0; y < img1_height; y++)
    {
        for (size_t x = 0; x < img1_width; x++)
        {
            float nx = (float)x / (img1_width - 1);
            float ny = (float)y / (img1_height - 1);
            size_t i = y * img1_width + x;
            float nv = img1_pixels[i] / 255.f;

            MAT_AT(td, i, 0) = nx;
            MAT_AT(td, i, 1) = ny;
            MAT_AT(td, i, 2) = 0.0f;
            MAT_AT(td, i, 3) = nv;
        }
    }

    for (size_t y = 0; y < img2_height; y++)
    {
        for (size_t x = 0; x < img2_width; x++)
        {
            float nx = (float)x / (img2_width - 1);
            float ny = (float)y / (img2_height - 1);
            size_t i = y * img2_width + x;
            size_t row = i + img1_height * img1_width;
            float nv = img2_pixels[i] / 255.f;

            MAT_AT(td, row, 0) = nx;
            MAT_AT(td, row, 1) = ny;
            MAT_AT(td, row, 2) = 1.0f;
            MAT_AT(td, row, 3) = nv;
        }
    }

    nn_rand(nn, -2, 2);

    int WINDOW_FACTOR = 80;
    int WINDOW_HEIGHT = 9 * WINDOW_FACTOR;
    int WINDOW_WIDTH = 16 * WINDOW_FACTOR;
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "gym");

    SetTargetFPS(60);

    size_t preview_image_height = img1_height * 3;
    size_t preview_image_width = img1_width * 3;

    Image preview_image = GenImageColor(preview_image_width, preview_image_height, BLACK);
    Texture2D preview_texture = LoadTextureFromImage(preview_image);

    float rate = 2;
    size_t batch_size = 28;
    size_t batch_count = (td.rows + batch_size - 1) / batch_size;
    size_t epoch_per_frame = 8;
    size_t epochs = 0;
    Cost_Plot cost_da = {0};
    size_t max_epoch = 100000;
    bool paused = false;

    float scroll = 0.5;
    bool scroll_dragging = false;

    while (!WindowShouldClose())
    {
        if (IsKeyPressed(KEY_SPACE))
        {
            paused = !paused;
        }
        if (IsKeyPressed(KEY_R))
        {
            epochs = 0;
            nn_rand(nn, -1, 1);
            cost_da.count = 0;
        }
        if (IsKeyPressed(KEY_S))
        {
            render_upscale_screenshot(nn, "upscale.png", scroll);
        }
        if (IsKeyPressed(KEY_X))
        {
            render_upscale_video(nn);
        }

        WINDOW_HEIGHT = GetRenderHeight();
        WINDOW_WIDTH = GetRenderWidth();
        float epoch_cost = 0;

        if (epochs < max_epoch)
        {
            for (size_t i = 0; i < epoch_per_frame && !paused; i++)
            {
                mat_shuffle_rows(td);
                for (size_t batch_current = 0; batch_current < batch_count; batch_current++)
                {
                    size_t training_size = batch_size;
                    size_t start_row = batch_size * batch_current;
                    if (start_row + training_size >= td.rows)
                    {
                        training_size = td.rows - start_row;
                    }

                    Mat batch_ti = {
                        .rows = training_size,
                        .cols = NN_INPUT(nn).cols,
                        .stride = td.stride,
                        .es = &MAT_AT(td, start_row, 0),
                    };

                    Mat batch_to = {
                        .rows = training_size,
                        .cols = NN_OUTPUT(nn).cols,
                        .stride = td.stride,
                        .es = &MAT_AT(td, start_row, batch_ti.cols),
                    };

                    nn_backprop(nn, g, batch_ti, batch_to);
                    nn_learn(nn, g, rate);

                    epoch_cost += nn_cost(nn, batch_ti, batch_to) / (batch_count * epoch_per_frame);
                }

                epochs++;
            }
        }

        BeginDrawing();
        Color background_color = {0x18, 0x18, 0x18, 0xFF};
        ClearBackground(background_color);

        char buf[256];
        snprintf(buf, sizeof(buf), "%zu: cost = %f\n", epochs, epoch_cost);
        if (epochs % 10 == 0)
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

        for (size_t y = 0; y < preview_image_height; y++)
        {
            for (size_t x = 0; x < preview_image_width; x++)
            {
                float nx = (float)x / (preview_image_width - 1);
                float ny = (float)y / (preview_image_height - 1);

                MAT_AT(NN_INPUT(nn), 0, 0) = nx;
                MAT_AT(NN_INPUT(nn), 0, 1) = ny;
                MAT_AT(NN_INPUT(nn), 0, 2) = scroll;
                nn_forward(nn);
                uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0) * 255.f;
                ImageDrawPixel(&preview_image, x, y, CLITERAL(Color){pixel, pixel, pixel, 255});
            }
        }
        UpdateTexture(preview_texture, preview_image.data);
        DrawTextureEx(preview_texture, CLITERAL(Vector2){x_offset, y_offset}, 0.f, 4, WHITE);

        {

            Vector2 size = {render_w * 0.85, render_h * 0.01};
            Vector2 rec_position = {x_offset, y_offset + render_h * 1};
            float radius = render_h * 0.05;

            DrawRectangleV(rec_position, size, WHITE);
            Vector2 knob_position = {rec_position.x + size.x * scroll, rec_position.y + size.y / 2};
            DrawCircleV(knob_position, radius, RED);

            if (scroll_dragging)
            {
                Vector2 mouse = GetMousePosition();
                float x_position = mouse.x;
                if (x_position < rec_position.x)
                {
                    x_position = rec_position.x + radius;
                }
                if (x_position > rec_position.x + size.x)
                {
                    x_position = rec_position.x + size.x - radius;
                }

                x_position -= rec_position.x;
                x_position = x_position / size.x;

                scroll = x_position;
            }

            if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
            {
                Vector2 mouse = GetMousePosition();
                if (Vector2Distance(mouse, knob_position) <= radius)
                {
                    scroll_dragging = true;
                }
            }
            if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT))
            {
                scroll_dragging = false;
            }
        }
        EndDrawing();
    }
    CloseWindow();

    return 0;

    return 0;
}
