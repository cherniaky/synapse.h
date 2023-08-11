#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include <raylib.h>

#define SYNAPSE_IMPLEMENTATION
#define SYNAPSE_ENABLE_GYM
#include "synapse.h"

void widget(Gym_Rect r, Color color)
{
    Rectangle rr = {.height = r.h, .width = r.w, .x = r.x, .y = r.y};
    if (CheckCollisionPointRec(GetMousePosition(), rr))
    {
        color = ColorBrightness(color, 0.4f);
    }

    DrawRectangleRec(rr, color);
}

int main(void)
{
    size_t factor = 80;
    size_t width = 16 * factor;
    size_t height = 9 * factor;

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(width, height, "Gym_Layout");
    SetTargetFPS(60);

    while (!WindowShouldClose())
    {
        int window_height = GetRenderHeight();
        int window_width = GetRenderWidth();
        size_t padding = window_height * 0.1;
        float gap = 10.f;

        BeginDrawing();
        ClearBackground(BLACK);

        gym_layout_begin(GLO_HORZ, gym_rect(0, padding, window_width, window_height - padding * 2), 3, gap);

        widget(gym_layout_slot(), RED);
        widget(gym_layout_slot(), GREEN);

        gym_layout_begin(GLO_VERT, gym_layout_slot(), 3, gap);

        widget(gym_layout_slot(), BLUE);
        gym_layout_begin(GLO_VERT, gym_layout_slot(), 3, gap);

        widget(gym_layout_slot(), BLUE);
        widget(gym_layout_slot(), YELLOW);
        widget(gym_layout_slot(), WHITE);

        gym_layout_end();
        widget(gym_layout_slot(), YELLOW);

        gym_layout_end();

        gym_layout_end();

        EndDrawing();
    }

    CloseWindow();

    return 0;
}