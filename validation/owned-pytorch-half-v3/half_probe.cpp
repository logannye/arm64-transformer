// Hosted-only differential probe of the original and candidate public headers.
// Binary output retains result bits and floating-point exception flags.
#include <torch/headeronly/util/Half.h>
#include <torch/headeronly/util/complex.h>

#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#pragma STDC FENV_ACCESS ON

using H = c10::Half;
using C = c10::complex<H>;
static_assert(sizeof(H) == 2 && alignof(H) == 2);
static_assert(sizeof(C) == 4 && alignof(C) == 4);
static_assert(std::is_same<decltype(H{} + H{}), H>::value);
static_assert(std::is_same<decltype(-H{}), H>::value);
static_assert(std::is_same<decltype(C{} += C{}), C &>::value);
static_assert(std::is_same<decltype(C{} *= C{}), C &>::value);

static H half(uint16_t bits) { return H(bits, H::from_bits()); }

static void word(uint32_t value) {
  if (std::fwrite(&value, sizeof(value), 1, stdout) != 1) {
    std::abort();
  }
}

static void result(H value) {
  const int flags = std::fetestexcept(FE_ALL_EXCEPT);
  word(value.x);
  word(static_cast<uint32_t>(flags));
}

static void result(C value) {
  const int flags = std::fetestexcept(FE_ALL_EXCEPT);
  word(value.real().x);
  word(value.imag().x);
  word(static_cast<uint32_t>(flags));
}

template <class F> static void record(F operation) {
  std::feclearexcept(FE_ALL_EXCEPT);
  result(operation());
}

int main(int argc, char **argv) {
  if (argc != 2 || !std::freopen(argv[1], "wb", stdout)) {
    return 1;
  }
  if (std::fesetround(FE_TONEAREST) != 0) {
    return 2;
  }
  // Zeros, smallest/largest subnormals, minimum normal, nearby units,
  // signed extremes, infinities, quiet/signaling NaNs and payload variants.
  const uint16_t partners[] = {0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3bff,
                               0x3c00, 0x3c01, 0xbc00, 0x7bff, 0xfbff, 0x7c00,
                               0xfc00, 0x7e00, 0x7c01, 0xfe55};
  for (uint32_t bits = 0; bits != 65536; ++bits) {
    const H a = half(static_cast<uint16_t>(bits));
    record([&] { return -a; });
    record([&] { return H(static_cast<float>(a)); });
    for (uint16_t other : partners) {
      const H b = half(other);
      record([&] { return a + b; });
      record([&] { return a - b; });
      record([&] { return a * b; });
      record([&] { return a / b; });
      const C initial(a, b);
      const C reversed(b, a);
      record([&] {
        auto c = initial;
        c += reversed;
        return c;
      });
      record([&] {
        auto c = initial;
        c -= reversed;
        return c;
      });
      record([&] {
        auto c = initial;
        c *= reversed;
        return c;
      });
      record([&] {
        auto c = initial;
        c += c;
        return c;
      });
      record([&] {
        auto c = initial;
        c -= c;
        return c;
      });
      record([&] {
        auto c = initial;
        c *= c;
        return c;
      });
      record([&] {
        return C(
            c10::complex<float>(static_cast<float>(a), static_cast<float>(b)));
      });
    }
    // Exact FP32 midpoint and its two adjacent floats exercise half rounding.
    if (bits < 0x7bff) {
      float low = static_cast<float>(a);
      float high = static_cast<float>(half(static_cast<uint16_t>(bits + 1)));
      float middle = (low + high) * 0.5f;
      for (float value : {std::nextafter(middle, low), middle,
                          std::nextafter(middle, high)}) {
        record([&] { return C(c10::complex<float>(value, -value)); });
      }
    }
  }
  return std::ferror(stdout) ? 3 : 0;
}
