#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"

#include "DeviceTable.h"
#include "HostTable.h"
#include "MultilevelTable.h"

int main() {
  wrapper();
  test();

  return 0;
}
