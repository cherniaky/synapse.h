#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include <raylib.h>

typedef struct
{
    float x;
    float y;
    float width;
    float height;
} Gym_Rect;

void widget(Gym_Rect r, Color color)
{
    Rectangle rr = {.height = r.height, .width = r.width, .x = r.x, .y = r.y};
    if (CheckCollisionPointRec(GetMousePosition(), rr))
    {
        color = ColorBrightness(color, 0.4f);
    }

    DrawRectangleRec(rr, color);
}

typedef enum
{
    GLO_HORZ,
    GLO_VERT,
} Gym_Layout_Orient;

typedef struct
{
    Gym_Layout_Orient orient;
    Gym_Rect rect;
    size_t elements_count;
    size_t i;
    float gap;
} Gym_Layout;

Gym_Layout make_layout(Gym_Layout_Orient orient, Gym_Rect rect, size_t count, float gap)
{
    Gym_Layout l = {
        .orient = orient,
        .rect = rect,
        .elements_count = count,
        .gap = gap,
    };
    return l;
}

Gym_Rect make_layout_rect(size_t x, size_t y, size_t width, size_t height)
{
    Gym_Rect rect = {
        .x = x,
        .y = y,
        .height = height,
        .width = width,
    };
    return rect;
}

Gym_Rect layout_slot(Gym_Layout *l, const char *filename, int line)
{
    if (l->i >= l->elements_count)
    {
        fprintf(stderr, "%s:%d: ERROR: Gym_Layout overflow\n", filename, line);
        exit(1);
    }

    Gym_Rect r = {0};
    switch (l->orient)
    {
    case GLO_HORZ:
        float width = l->rect.width / l->elements_count;
        r.x = l->rect.x + l->i * width;
        r.y = l->rect.y;
        r.width = width;
        r.height = l->rect.height;

        if (l->i == 0)
        {
            r.width -= l->gap / 2;
        }
        else if (l->i == l->elements_count - 1)
        {
            r.x += l->gap / 2;
            r.width -= l->gap / 2;
        }
        else
        {
            r.x += l->gap / 2;
            r.width -= l->gap;
        }

        break;

    case GLO_VERT:
        float height = l->rect.height / l->elements_count;
        r.x = l->rect.x;
        r.y = l->rect.y + l->i * height;
        r.width = l->rect.width;
        r.height = height;

        if (l->i == 0)
        {
            r.height -= l->gap / 2;
        }
        else if (l->i == l->elements_count - 1)
        {
            r.y += l->gap / 2;
            r.height -= l->gap / 2;
        }
        else
        {
            r.y += l->gap / 2;
            r.height -= l->gap;
        }

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
    Gym_Layout *items;
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

void gym_layout_stack_push(Layout_Stack *ls, Gym_Layout_Orient orient, Gym_Rect rect, size_t count, float padding)
{
    Gym_Layout l = make_layout(orient, rect, count, padding);
    da_append(ls, l);
}
void layout_stack_pop(Layout_Stack *ls)
{
    assert(ls->count > 0);
    ls->count--;
}

Gym_Rect gym_layout_stack_slot_loc(Layout_Stack *ls, const char *filename, int line)
{
    assert(ls->count > 0);
    return layout_slot(&ls->items[ls->count - 1], filename, line);
}

#define gym_layout_stack_slot(ls) gym_layout_stack_slot_loc(ls, __FILE__, __LINE__)

int main(void)
{
    size_t factor = 80;
    size_t width = 16 * factor;
    size_t height = 9 * factor;

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(width, height, "Gym_Layout");
    SetTargetFPS(60);

    Layout_Stack ls = {0};

    while (!WindowShouldClose())
    {
        int window_height = GetRenderHeight();
        int window_width = GetRenderWidth();
        size_t padding = window_height * 0.1;
        float gap = 10.f;

        BeginDrawing();
        ClearBackground(BLACK);

        gym_layout_stack_push(&ls, GLO_HORZ, make_layout_rect(0, padding, window_width, window_height - padding * 2), 3, gap);

        widget(gym_layout_stack_slot(&ls), RED);
        widget(gym_layout_stack_slot(&ls), GREEN);

        gym_layout_stack_push(&ls, GLO_VERT, gym_layout_stack_slot(&ls), 3, gap);

        widget(gym_layout_stack_slot(&ls), BLUE);
        gym_layout_stack_push(&ls, GLO_VERT, gym_layout_stack_slot(&ls), 3, gap);

        widget(gym_layout_stack_slot(&ls), BLUE);
        widget(gym_layout_stack_slot(&ls), YELLOW);
        widget(gym_layout_stack_slot(&ls), WHITE);

        layout_stack_pop(&ls);
        widget(gym_layout_stack_slot(&ls), YELLOW);

        layout_stack_pop(&ls);

        layout_stack_pop(&ls);

        EndDrawing();
        assert(ls.count == 0);
    }

    CloseWindow();

    return 0;
}