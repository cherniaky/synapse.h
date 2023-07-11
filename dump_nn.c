#define SYNAPSE_IMPLEMENTATION ;
#include "synapse.h";
#define OLIVEC_IMPLEMENTATION ;
#include "./dev_deps/olive.c";
#define STB_IMAGE_WRITE_IMPLEMENTATION ;
#include "./dev_deps/stb_image_write.h";

#define IMG_WIDTH 800
#define IMG_HEIGHT 600

uint32_t img_pixels[IMG_HEIGHT * IMG_WIDTH];

int main(void)
{
    size_t arch[] = {4, 4, 2, 1};
    int arch_count = ARRAY_LEN(arch);
    NN nn = nn_alloc(arch, arch_count);
    nn_rand(nn, -1, 1);

    NN_PRINT(nn);

    uint32_t neuron_color = 0xFF0000FF;
    uint32_t background_color = 0xFF181818;
    Olivec_Canvas img = olivec_canvas(img_pixels, IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH);

    olivec_fill(img, background_color);

    for (size_t j = 0; j < arch_count; j++)
    {
        int layer_count = arch[j];
        int neuron_radius = 20;
        int layer_border_pad = 20;

        int layer_height = img.height - 2 * layer_border_pad;
        int nn_width = img.width - 2 * layer_border_pad;

        int layer_vpad = layer_height / (layer_count + 1);
        int layer_hpad = nn_width / (arch_count + 1);

        int layer_x = img.width / 2 - nn_width / 2;
        int layer_y = img.height / 2 - layer_height / 2;

        for (size_t i = 0; i < layer_count; i++)
        {
            int cx = layer_x + (j + 1) * layer_hpad;
            int cy = layer_y + (i + 1) * layer_vpad;
            olivec_circle(img, cx, cy, neuron_radius, neuron_color);
        }
    }

    const char *img_file_path = "nn.png";
    if (!stbi_write_png(img_file_path, img.width, img.height, 4, img.pixels, img.stride * sizeof(uint32_t)))
    {
        printf("Error: could not save file to %s\n", img_file_path);
        return 1;
    }

    printf("Saved file to %s\n", img_file_path);

    return 0;
}