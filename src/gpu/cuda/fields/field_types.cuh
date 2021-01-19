#ifndef FIELD_TYPES_HEADER
#define FIELD_TYPES_HEADER

#define cu_func __device__ __noinline__
#define cu_inline_func __device__ __forceinline__
#define cu_entry_func __global__

#include <assert.h>
#include <climits>
#include <type_traits>

#include "field_structs.hpp"

// Returns a * b + c + d, puts the carry in d
static cu_inline_func uint64_t mac_with_carry_64(uint64_t a, uint64_t b, uint64_t c, uint64_t *d) {
    uint64_t lo, hi;
    asm("mad.lo.cc.u64 %0, %2, %3, %4;\r\n"
        "madc.hi.u64 %1, %2, %3, 0;\r\n"
        "add.cc.u64 %0, %0, %5;\r\n"
        "addc.u64 %1, %1, 0;\r\n"
        : "=l"(lo), "=l"(hi) : "l"(a), "l"(b), "l"(c), "l"(*d));
    *d = hi;
    return lo;
}
  
// Returns a + b, puts the carry in d
static cu_inline_func uint64_t add_with_carry_64(uint64_t a, uint64_t *b) {
  uint64_t lo, hi;
  asm("add.cc.u64 %0, %2, %3;\r\n"
      "addc.u64 %1, 0, 0;\r\n"
      : "=l"(lo), "=l"(hi) : "l"(a), "l"(*b));
  *b = hi;
  return lo;
}

// Returns a * b + c + d, puts the carry in d
static cu_inline_func uint32_t mac_with_carry_32(uint32_t a, uint32_t b, uint32_t c, uint32_t *d) {
  uint64_t res = (uint64_t)a * b + c + *d;
  *d = res >> 32;
  return res;
}

// Returns a + b, puts the carry in b
static cu_inline_func uint32_t add_with_carry_32(uint32_t a, uint32_t *b) {
    uint32_t lo, hi;
    asm("add.cc.u32 %0, %2, %3;\r\n"
        "addc.u32 %1, 0, 0;\r\n"
        : "=r"(lo), "=r"(hi) : "r"(a), "r"(*b));
    *b = hi;
    return lo;
}

cu_inline_func limb_type Fr::add_with_carry(limb_type a, limb_type *b) {
    static_assert(std::is_same<Fr::limb_type, uint64_t>::value, "unmatched types");
    return add_with_carry_64(a, b);
}

cu_inline_func
limb_type Fr::mac_with_carry(Fr::limb_type a,
                             Fr::limb_type b,
                             Fr::limb_type c,
                             Fr::limb_type *d) {
    static_assert(std::is_same<Fr::limb_type, uint64_t>::value, "type mismatch");
    return mac_with_carry_64(a, b, c, d);
}

cu_inline_func
Fr Fr::add_(Fr a, Fr b) {
    static_assert(std::is_same<Fr::limb_type, uint64_t>::value, "unmatched types");
    asm("add.cc.u64 %0, %0, %4;\r\n"
    "addc.cc.u64 %1, %1, %5;\r\n"
    "addc.cc.u64 %2, %2, %6;\r\n"
    "addc.u64 %3, %3, %7;\r\n"
    :"+l"(a.val[0]), "+l"(a.val[1]), "+l"(a.val[2]), "+l"(a.val[3])
    :"l"(b.val[0]), "l"(b.val[1]), "l"(b.val[2]), "l"(b.val[3]));
    return a;
}

cu_inline_func 
Fr Fr::sub_(Fr a, Fr b) {
    static_assert(std::is_same<Fr::limb_type, uint64_t>::value, "unmatched types");
    asm("sub.cc.u64 %0, %0, %4;\r\n"
        "subc.cc.u64 %1, %1, %5;\r\n"
        "subc.cc.u64 %2, %2, %6;\r\n"
        "subc.u64 %3, %3, %7;\r\n"
        :"+l"(a.val[0]), "+l"(a.val[1]), "+l"(a.val[2]), "+l"(a.val[3])
        :"l"(b.val[0]), "l"(b.val[1]), "l"(b.val[2]), "l"(b.val[3]));
    return a;
}

// TODO:
const Fr Fr::ONE = { 8589934590ull,
                   6378425256633387010ull,
                   11064306276430008309ull,
                   1739710354780652911ull };
const Fr Fr::P = { 18446744069414584321ull,
                   6034159408538082302ull,
                   3691218898639771653ull,
                   8353516859464449352ull };
const Fr Fr::ZERO = {0, 0, 0, 0};
const Fr Fr::R2 = { 14526898881837571181ull,
                    3129137299524312099ull,
                    419701826671360399ull,
                    524908885293268753ull };
const Fr::limb_type Fr::INV = 18446744069414584319ull;

const Fq Fq::P = { 13402431016077863595ull,
                   2210141511517208575ull,
                   7435674573564081700ull,
                   7239337960414712511ull,
                   5412103778470702295ull,
                   1873798617647539866ull };
const Fq Fq::ZERO = {0, 0, 0, 0, 0, 0};
const Fq Fq::R2 = { 17644856173732828998ull,
                    754043588434789617ull,
                    10224657059481499349ull,
                    7488229067341005760ull,
                    11130996698012816685ull,
                    1267921511277847466ull };
const Fq Fq::ONE = { 8505329371266088957ull,
                     17002214543764226050ull,
                     6865905132761471162ull,
                     8632934651105793861ull,
                     6631298214892334189ull,
                     1582556514881692819ull };
const Fq::limb_type Fq::INV = 9940570264628428797ull;

cu_inline_func
limb_type Fq::add_with_carry(limb_type a, limb_type *b) {
    static_assert(std::is_same<Fr::limb_type, uint64_t>::value, "unmatched types");
    return add_with_carry_64(a, b);
}

cu_inline_func
limb_type Fq::mac_with_carry(Fr::limb_type a,
                             Fr::limb_type b,
                             Fr::limb_type c,
                             Fr::limb_type *d) {
    static_assert(std::is_same<Fr::limb_type, uint64_t>::value, "type mismatch");
    return mac_with_carry_64(a, b, c, d);
}

cu_inline_func
Fq Fq::sub_(Fq a, Fq b) {
    static_assert(std::is_same<Fr::limb_type, uint64_t>::value, "unmatched types");
    asm("sub.cc.u64 %0, %0, %6;\r\n"
    "subc.cc.u64 %1, %1, %7;\r\n"
    "subc.cc.u64 %2, %2, %8;\r\n"
    "subc.cc.u64 %3, %3, %9;\r\n"
    "subc.cc.u64 %4, %4, %10;\r\n"
    "subc.u64 %5, %5, %11;\r\n"
    :"+l"(a.val[0]), "+l"(a.val[1]), "+l"(a.val[2]), "+l"(a.val[3]), "+l"(a.val[4]), "+l"(a.val[5])
    :"l"(b.val[0]), "l"(b.val[1]), "l"(b.val[2]), "l"(b.val[3]), "l"(b.val[4]), "l"(b.val[5]));
    return a;
}

cu_inline_func
Fq Fq::add_(Fq a, Fq b) {
    static_assert(std::is_same<Fr::limb_type, uint64_t>::value, "unmatched types");
    asm("add.cc.u64 %0, %0, %6;\r\n"
        "addc.cc.u64 %1, %1, %7;\r\n"
        "addc.cc.u64 %2, %2, %8;\r\n"
        "addc.cc.u64 %3, %3, %9;\r\n"
        "addc.cc.u64 %4, %4, %10;\r\n"
        "addc.u64 %5, %5, %11;\r\n"
        :"+l"(a.val[0]), "+l"(a.val[1]), "+l"(a.val[2]), "+l"(a.val[3]), "+l"(a.val[4]), "+l"(a.val[5])
        :"l"(b.val[0]), "l"(b.val[1]), "l"(b.val[2]), "l"(b.val[3]), "l"(b.val[4]), "l"(b.val[5]));
    return a;
}

const Fq2 Fq2::ONE = {Fq::ONE, Fq::ZERO};
const Fq2 Fq2::ZERO = {Fq::ZERO, Fq::ZERO};

template<>
const projective<Fq>  projective<Fq>::ZERO = {Fq::ZERO, Fq::ONE, Fq::ZERO};

template<>
const projective<Fq2> projective<Fq2>::ZERO = {Fq2::ZERO, Fq2::ONE, Fq2::ZERO};

#endif
