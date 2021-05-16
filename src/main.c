#include <stdio.h>
#include <stdlib.h>

#include "cuckoo.h"
#include "xxhash.h"
#include "types.h"

int main() {
  table *t = table_new(10, 2);

  for (usize i = 0; i < 10; i += 1) {
    table_insert(t, rand() % 0xff);
    table_write(t, stdout);
    putchar('\n');
  }

  table_free(t);

  return 0;
}
