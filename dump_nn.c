#include <time.h>

#define SYNAPSE_IMPLEMENTATION
#include "synapse.h"
#define OLIVEC_IMPLEMENTATION
#include "./dev_deps/olive.c"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./dev_deps/stb_image_write.h"

#define IMG_WIDTH 800
#define IMG_HEIGHT 600

uint32_t img_pixels[IMG_HEIGHT * IMG_WIDTH];

void nn_render(Olivec_Canvas img, NN nn, size_t *arch)
{
    uint32_t background_color = 0xFF181818;
    uint32_t low = 0xFF0000FF;
    uint32_t high = 0x00FFFF00;

    olivec_fill(img, background_color);
    int arch_count = nn.count + 1;

    for (size_t l = 0; l < arch_count; l++)
    {
        int layer_count = arch[l];
        int neuron_radius = 20;
        int layer_border_pad = 20;

        int layer_height = img.height - 2 * layer_border_pad;
        int nn_width = img.width - 2 * layer_border_pad;

        int layer_vpad = layer_height / (layer_count + 1);
        int nn_hpad = nn_width / (arch_count + 1);

        int nn_x = img.width / 2 - nn_width / 2;
        int nn_y = img.height / 2 - layer_height / 2;

        for (size_t i = 0; i < layer_count; i++)
        {
            int cx1 = nn_x + (l + 1) * nn_hpad;
            int cy1 = nn_y + (i + 1) * layer_vpad;

            if (l < arch_count - 1)
            {
                for (size_t j = 0; j < arch[l + 1]; j++)
                {
                    int cx2 = nn_x + (l + 1 + 1) * nn_hpad;
                    int next_layer_vpad = layer_height / (arch[l + 1] + 1);
                    int cy2 = nn_y + (j + 1) * next_layer_vpad;

                    uint32_t connection_color = low;
                    uint32_t alpha = floorf(sigmoidf(MAT_AT(nn.ws[l], i, j)) * 255.f);
                    olivec_blend_color(&connection_color, (alpha << (8 * 3)) | high);
                    olivec_line(img, cx1, cy1, cx2, cy2, connection_color);
                }
            }
            uint32_t neuron_color = low;
            if (l == 0)
            {
                neuron_color = 0xff505050;
            }
            else
            {
                uint32_t alpha = floorf(sigmoidf(MAT_AT(nn.bs[l - 1], 0, i)) * 255.f);
                olivec_blend_color(&neuron_color, (alpha << (8 * 3)) | high);
            }
            olivec_circle(img, cx1, cy1, neuron_radius, neuron_color);
        }
    }
}

int main(void)
{
    srand(time(0));
    size_t arch[] = {4, 4, 2, 1};
    int arch_count = ARRAY_LEN(arch);
    NN nn = nn_alloc(arch, arch_count);
    nn_rand(nn, -1, 1);

    NN_PRINT(nn);

    Olivec_Canvas img = olivec_canvas(img_pixels, IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH);

    nn_render(img, nn, arch);

    const char *img_file_path = "nn.png";
    if (!stbi_write_png(img_file_path, img.width, img.height, 4, img.pixels, img.stride * sizeof(uint32_t)))
    {
        printf("Error: could not save file to %s\n", img_file_path);
        return 1;
    }

    printf("Saved file to %s\n", img_file_path);

    return 0;
}