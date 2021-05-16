#include <stdlib.h>
#include "types.h"
#include "hasher.h"

void hasher_init(hasher *h, u32 mask) {
    h->r = rand();
    h->x = rand();
    h->m = rand();
    h->mask = mask;
}

u32 hasher_hash(hasher *h, i32 v) {
    u32 a = ((u32)(v) << h->r) | ((u32)(v) >> h->r);
    u32 b = a ^ h->x;
    u32 c = b * h->m;
    return c;
}
