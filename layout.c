#include <stdio.h>
#include <assert.h>

typedef struct
{
    float x;
    float y;
    float w;
    float h;
} Layout_Rect;

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
} Layout;

Layout_Rect layout_slot(Layout *l, size_t i)
{
    assert(i < l->elements_count);

    Layout_Rect r = {0};
    switch (l->orient)
    {
    case LO_HORZ:
        float w = l->rect.w / l->elements_count;
        r.x = l->rect.x + i * w;
        r.y = l->rect.y;
        r.w = w;
        r.h = l->rect.h;

        break;

    case LO_VERT:
        float h = l->rect.h / l->elements_count;
        r.x = l->rect.x;
        r.y = l->rect.y + i * h;
        r.w = l->rect.w;
        r.h = h;

        break;
    default:
        assert(0 && "unreachable");
        break;
    }

    return r;
}

int main(void)
{
    size_t width = 1920;
    size_t height = 1080;

    Layout root = {
        .orient = LO_HORZ,
        .rect = {0, 0, width, height},
        .elements_count = 3,
    };

    layout_slot(&root, 0);

    return 0;
}