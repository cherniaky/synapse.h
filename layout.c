#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include <raylib.h>

typedef struct
{
    float x;
    float y;
    float w;
    float h;
} Layout_Rect;

void widget(Layout_Rect r, Color color)
{
    DrawRectangle(ceilf(r.x), ceilf(r.y), ceilf(r.w), ceilf(r.h), color);
}

typedef enum
{
    LO_HORZ,
    LO_VERT,
} Layout_Orient;

typedef struct
{
    Layout_Orient orient;
    Layout_Rect rect;
    size_t elements_count;
    size_t i;
} Layout;

Layout make_layout(Layout_Orient orient, Layout_Rect rect, size_t count)
{
    Layout l = {
        .orient = orient,
        .rect = rect,
        .elements_count = count,
    };
    return l;
}

Layout_Rect make_layout_rect(size_t x, size_t y, size_t w, size_t h)
{
    Layout_Rect rect = {
        .x = x,
        .y = y,
        .h = h,
        .w = w,
    };
    return rect;
}

Layout_Rect layout_slot(Layout *l)
{
    assert(l->i < l->elements_count);

    Layout_Rect r = {0};
    switch (l->orient)
    {
    case LO_HORZ:
        float w = l->rect.w / l->elements_count;
        r.x = l->rect.x + l->i * w;
        r.y = l->rect.y;
        r.w = w;
        r.h = l->rect.h;

        break;

    case LO_VERT:
        float h = l->rect.h / l->elements_count;
        r.x = l->rect.x;
        r.y = l->rect.y + l->i * h;
        r.w = l->rect.w;
        r.h = h;

        break;
    default:
        assert(0 && "unreachable");
        break;
    }

    l->i += 1;

    return r;
}

typedef struct
{
    Layout *items;
    size_t count;
    size_t capacity;
} Layout_Stack;

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

void layout_stack_push(Layout_Stack *ls, Layout_Orient orient, Layout_Rect rect, size_t count)
{
    Layout l = make_layout(orient, rect, count);
    da_append(ls, l);
}
void layout_stack_pop(Layout_Stack *ls)
{
    assert(ls->count > 0);
    ls->count--;
}

Layout_Rect layout_stack_slot_rect(Layout_Stack *ls)
{
    assert(ls->count > 0);
    return layout_slot(&ls->items[ls->count - 1]);
}

int main(void)
{
    size_t factor = 80;
    size_t width = 16 * factor;
    size_t height = 9 * factor;

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(width, height, "Layout");
    SetTargetFPS(60);

    Layout_Stack ls = {0};

    while (!WindowShouldClose())
    {
        int window_height = GetRenderHeight();
        int window_width = GetRenderWidth();
        size_t padding = window_height * 0.1;

        BeginDrawing();
        ClearBackground(BLACK);

        layout_stack_push(&ls, LO_HORZ, make_layout_rect(0, padding, window_width, window_height - padding * 2), 3);

        widget(layout_stack_slot_rect(&ls), RED);
        widget(layout_stack_slot_rect(&ls), GREEN);

        layout_stack_push(&ls, LO_VERT, layout_stack_slot_rect(&ls), 3);

        widget(layout_stack_slot_rect(&ls), BLUE);
        widget(layout_stack_slot_rect(&ls), YELLOW);
        widget(layout_stack_slot_rect(&ls), MAGENTA);

        layout_stack_pop(&ls);

        layout_stack_pop(&ls);

        EndDrawing();
        assert(ls.count == 0);
    }

    CloseWindow();

    return 0;
}