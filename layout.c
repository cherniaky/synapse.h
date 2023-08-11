#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include <raylib.h>

// typedef struct
// {
//     float x;
//     float y;
//     float width;
//     float height;
// } Rectangle;

void widget(Rectangle r, Color color)
{
    if (CheckCollisionPointRec(GetMousePosition(), r))
    {
        color = ColorBrightness(color, 0.4f);
    }

    DrawRectangleRec(r, color);
}

typedef enum
{
    LO_HORZ,
    LO_VERT,
} Layout_Orient;

typedef struct
{
    Layout_Orient orient;
    Rectangle rect;
    size_t elements_count;
    size_t i;
} Layout;

Layout make_layout(Layout_Orient orient, Rectangle rect, size_t count)
{
    Layout l = {
        .orient = orient,
        .rect = rect,
        .elements_count = count,
    };
    return l;
}

Rectangle make_layout_rect(size_t x, size_t y, size_t width, size_t height)
{
    Rectangle rect = {
        .x = x,
        .y = y,
        .height = height,
        .width = width,
    };
    return rect;
}

Rectangle layout_slot(Layout *l)
{
    assert(l->i < l->elements_count);

    Rectangle r = {0};
    switch (l->orient)
    {
    case LO_HORZ:
        float width = l->rect.width / l->elements_count;
        r.x = l->rect.x + l->i * width;
        r.y = l->rect.y;
        r.width = width;
        r.height = l->rect.height;

        break;

    case LO_VERT:
        float height = l->rect.height / l->elements_count;
        r.x = l->rect.x;
        r.y = l->rect.y + l->i * height;
        r.width = l->rect.width;
        r.height = height;

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

void layout_stack_push(Layout_Stack *ls, Layout_Orient orient, Rectangle rect, size_t count)
{
    Layout l = make_layout(orient, rect, count);
    da_append(ls, l);
}
void layout_stack_pop(Layout_Stack *ls)
{
    assert(ls->count > 0);
    ls->count--;
}

Rectangle layout_stack_slot_rect(Layout_Stack *ls)
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
        layout_stack_push(&ls, LO_VERT, layout_stack_slot_rect(&ls), 3);

        widget(layout_stack_slot_rect(&ls), BLUE);
        widget(layout_stack_slot_rect(&ls), YELLOW);
        widget(layout_stack_slot_rect(&ls), WHITE);

        layout_stack_pop(&ls);
        widget(layout_stack_slot_rect(&ls), YELLOW);

        layout_stack_pop(&ls);

        layout_stack_pop(&ls);

        EndDrawing();
        assert(ls.count == 0);
    }

    CloseWindow();

    return 0;
}