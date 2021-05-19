#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"

#include "DeviceTable.h"
#include "HostTable.h"
#include "Types.h"
#include "xxHash.h"

int main() {
  auto t = HostTable(10, 2);

  for (usize i = 0; i < 10; i += 1) {
    t.insert(rand());
    t.write(stdout);
    putchar('\n');
  }

  wrapper();

  return 0;
}
