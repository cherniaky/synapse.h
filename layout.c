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

    Gym_Layout_Stack ls = {0};

    while (!WindowShouldClose())
    {
        int window_height = GetRenderHeight();
        int window_width = GetRenderWidth();
        size_t padding = window_height * 0.1;
        float gap = 10.f;

        BeginDrawing();
        ClearBackground(BLACK);

        gls_push(&ls, GLO_HORZ, gym_rect(0, padding, window_width, window_height - padding * 2), 3, gap);

        widget(gls_slot(&ls), RED);
        widget(gls_slot(&ls), GREEN);

        gls_push(&ls, GLO_VERT, gls_slot(&ls), 3, gap);

        widget(gls_slot(&ls), BLUE);
        gls_push(&ls, GLO_VERT, gls_slot(&ls), 3, gap);

        widget(gls_slot(&ls), BLUE);
        widget(gls_slot(&ls), YELLOW);
        widget(gls_slot(&ls), WHITE);

        gls_pop(&ls);
        widget(gls_slot(&ls), YELLOW);

        gls_pop(&ls);

        gls_pop(&ls);

        EndDrawing();
        assert(ls.count == 0);
    }

    CloseWindow();

    return 0;
}