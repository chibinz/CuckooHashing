#include <stdio.h>
#include <stdlib.h>

#include "HostTable.h"
#include "xxHash.h"
#include "Types.h"

int main() {
  auto t = HostTable(10, 2);

  for (usize i = 0; i < 10; i += 1) {
    t.insert(rand() % 0xff);
    t.write(stdout);
    putchar('\n');
  }

  return 0;
}
