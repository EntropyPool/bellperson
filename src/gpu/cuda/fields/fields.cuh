#ifndef FIELDS_HEADER
#define FIELDS_HEADER

#include "field_types.cuh"

#include <stdio.h>

// CUDA-specific constants
#define CUDA_CONSTANTS(T) \
  static __constant__ T T##_ONE; \
  static __constant__ T T##_ZERO;\
  static __constant__ T T##_P;   \
  static __constant__ T T##_R2 


CUDA_CONSTANTS(Fr);
CUDA_CONSTANTS(Fq);
CUDA_CONSTANTS(Fq2);

static __constant__ projective<Fq>  G1_ONE;
static __constant__ projective<Fq>  G1_ZERO;

static __constant__ projective<Fq2> G2_ONE;
static __constant__ projective<Fq2> G2_ZERO;

#define GET_ZERO(T) (T##_ZERO)
#define GET_ONE(T) (T##_ONE)
#define GET_P(T) (T##_P)
#define GET_R2(T) (T##_R2)

#define DECLARE_CONSTANT_GETTER(T, N) \
  cu_func T T::Get_##N() {            \
    return T##_##N;                   \
  }

#define DECLARE_CONSTANT_PTR_GETTER(T, N) \
  cu_func T *T::Get_##N##_PTR() {            \
    return &T##_##N;                   \
  }

DECLARE_CONSTANT_GETTER(Fq, ONE);
DECLARE_CONSTANT_GETTER(Fq, P);
DECLARE_CONSTANT_GETTER(Fq, ZERO);
DECLARE_CONSTANT_GETTER(Fq, R2);

DECLARE_CONSTANT_GETTER(Fr, ONE);
DECLARE_CONSTANT_GETTER(Fr, P);
DECLARE_CONSTANT_GETTER(Fr, ZERO);
DECLARE_CONSTANT_GETTER(Fr, R2);

DECLARE_CONSTANT_GETTER(Fq2, ONE);
DECLARE_CONSTANT_GETTER(Fq2, ZERO);

DECLARE_CONSTANT_PTR_GETTER(Fq, ONE);
DECLARE_CONSTANT_PTR_GETTER(Fq, P);
DECLARE_CONSTANT_PTR_GETTER(Fq, ZERO);
DECLARE_CONSTANT_PTR_GETTER(Fq, R2);

DECLARE_CONSTANT_PTR_GETTER(Fr, ONE);
DECLARE_CONSTANT_PTR_GETTER(Fr, P);
DECLARE_CONSTANT_PTR_GETTER(Fr, ZERO);
DECLARE_CONSTANT_PTR_GETTER(Fr, R2);

DECLARE_CONSTANT_PTR_GETTER(Fq2, ONE);
DECLARE_CONSTANT_PTR_GETTER(Fq2, ZERO);

template<>
cu_inline_func projective<Fq> projective<Fq>::Get_ZERO() {
  return G1_ZERO;
}

template<>
cu_inline_func projective<Fq2> projective<Fq2>::Get_ZERO() {
  return G2_ZERO;
}

template<>
cu_inline_func projective<Fq> *projective<Fq>::Get_ZERO_PTR() {
  return &G1_ZERO;
}

template<>
cu_inline_func projective<Fq2> *projective<Fq2>::Get_ZERO_PTR() {
  return &G2_ZERO;
}

template<>
cu_inline_func void projective<Fq>::COPY_ZERO(projective<Fq> *d) {
  memcpy(d, &G1_ZERO, sizeof(G1_ZERO));
}

template<>
cu_inline_func void projective<Fq2>::COPY_ZERO(projective<Fq2> *d) {
  memcpy(d, &G2_ZERO, sizeof(G2_ZERO));
}

void instantiate_constants() {
#define INSTANTIATE_CONSTANT(T, N) \
  CUDA_CHECK(cudaMemcpyToSymbolAsync(T##_##N, &T::N, sizeof(T)))

  INSTANTIATE_CONSTANT(Fr, ONE);
  INSTANTIATE_CONSTANT(Fr, ZERO);
  INSTANTIATE_CONSTANT(Fr, P);
  INSTANTIATE_CONSTANT(Fr, R2);

  INSTANTIATE_CONSTANT(Fq, ONE);
  INSTANTIATE_CONSTANT(Fq, ZERO);
  INSTANTIATE_CONSTANT(Fq, P);
  INSTANTIATE_CONSTANT(Fq, R2);

  INSTANTIATE_CONSTANT(Fq2, ONE);
  INSTANTIATE_CONSTANT(Fq2, ZERO);

  CUDA_CHECK(cudaMemcpyToSymbol(G1_ZERO, &projective<Fq>::ZERO, sizeof(projective<Fq>)));
  CUDA_CHECK(cudaMemcpyToSymbol(G2_ZERO, &projective<Fq2>::ZERO, sizeof(projective<Fq2>)));
}

// Greater than or equal
template<typename FIELD>
static cu_inline_func bool gte(FIELD a, FIELD b) {
  for(int i = FIELD::FIELD_LIMBS - 1; i >= 0; i--){
    if(a.val[i] > b.val[i])
      return true;
    if(a.val[i] < b.val[i])
      return false;
  }
  return true;
}

// Equals
template<typename FIELD>
static cu_inline_func bool eq(FIELD a, FIELD b) {
  for(unsigned i = 0; i < FIELD::FIELD_LIMBS; i++)
    if(a.val[i] != b.val[i])
      return false;
  return true;
}

template<> cu_inline_func bool eq(Fq2 a, Fq2 b) {
  return eq(a.c0, b.c0) && eq(a.c1, b.c1);
}

// Squaring is a special case of multiplication which can be done ~1.5x faster.
// https://stackoverflow.com/a/16388571/1348497
template<typename FIELD>
static cu_inline_func FIELD sqr(FIELD a) {
  return mul(a, a);
}

// Modular subtraction
template<typename FIELD>
static cu_inline_func FIELD sub(FIELD a, FIELD b) {
  FIELD res = FIELD::sub_(a, b);
  if(!gte(a, b)) res = FIELD::add_(res, *FIELD::Get_P_PTR());
  return res;
}

template<> cu_inline_func Fq2 sub(Fq2 a, Fq2 b) {
  a.c0 = sub(a.c0, b.c0);
  a.c1 = sub(a.c1, b.c1);
  return a;
}

// Modular addition
template<typename FIELD>
static cu_inline_func FIELD add(FIELD a, FIELD b) {
  FIELD res = FIELD::add_(a, b);
  if(gte(res, *FIELD::Get_P_PTR())) res = FIELD::sub_(res, *FIELD::Get_P_PTR());
  return res;
}

template<> cu_inline_func Fq2 add(Fq2 a, Fq2 b) {
  a.c0 = add(a.c0, b.c0);
  a.c1 = add(a.c1, b.c1);
  return a;
}

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
template<typename G>
static cu_func projective<G> add(projective<G> a, projective<G> b) {
  if(eq(a.z, *G::Get_ZERO_PTR())) return b;
  if(eq(b.z, *G::Get_ZERO_PTR())) return a;

  const G z1z1 = sqr(a.z); // Z1Z1 = Z1^2
  const G z2z2 = sqr(b.z); // Z2Z2 = Z2^2
  const G u1 = mul(a.x, z2z2); // U1 = X1*Z2Z2
  const G u2 = mul(b.x, z1z1); // U2 = X2*Z1Z1
  G s1 = mul(mul(a.y, b.z), z2z2); // S1 = Y1*Z2*Z2Z2
  const G s2 = mul(mul(b.y, a.z), z1z1); // S2 = Y2*Z1*Z1Z1

  if(eq(u1, u2) && eq(s1, s2))
    return double_op(a);
  else {
    const G h = sub(u2, u1); // H = U2-U1
    G i = double_op(h); i = sqr(i); // I = (2*H)^2
    const G j = mul(h, i); // J = H*I
    G r = sub(s2, s1); r = double_op(r); // r = 2*(S2-S1)
    const G v = mul(u1, i); // V = U1*I
    a.x = sub(sub(sub(sqr(r), j), v), v); // X3 = r^2 - J - 2*V

    // Y3 = r*(V - X3) - 2*S1*J
    a.y = mul(sub(v, a.x), r);
    s1 = mul(s1, j); s1 = double_op(s1); // S1 = S1 * J * 2
    a.y = sub(a.y, s1);

    // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
    a.z = add(a.z, b.z); a.z = sqr(a.z);
    a.z = sub(sub(a.z, z1z1), z2z2);
    a.z = mul(a.z, h);

    return a;
  }
}

// Modular multiplication
template<typename FIELD>
static cu_func FIELD mul(FIELD a, FIELD b) {
  /* CIOS Montgomery multiplication, inspired from Tolga Acar's thesis:
   * https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf
   * Learn more:
   * https://en.wikipedia.org/wiki/Montgomery_modular_multiplication
   * https://alicebob.cryptoland.net/understanding-the-montgomery-reduction-algorithm/
   */
  typename FIELD::limb_type t[FIELD::FIELD_LIMBS + 2] = {0};
  for(unsigned i = 0; i < FIELD::FIELD_LIMBS; i++) {
    typename FIELD::limb_type carry = 0;
    for(unsigned j = 0; j < FIELD::FIELD_LIMBS; j++)
      t[j] = FIELD::mac_with_carry(a.val[j], b.val[i], t[j], &carry);
    t[FIELD::FIELD_LIMBS] = FIELD::add_with_carry(t[FIELD::FIELD_LIMBS], &carry);
    t[FIELD::FIELD_LIMBS + 1] = carry;

    carry = 0;
    typename FIELD::limb_type m = FIELD::INV * t[0];
    FIELD::mac_with_carry(m, FIELD::Get_P_PTR()->val[0], t[0], &carry);
    for(unsigned j = 1; j < FIELD::FIELD_LIMBS; j++)
      t[j - 1] = FIELD::mac_with_carry(m, FIELD::Get_P_PTR()->val[j], t[j], &carry);

    t[FIELD::FIELD_LIMBS - 1] = FIELD::add_with_carry(t[FIELD::FIELD_LIMBS], &carry);
    t[FIELD::FIELD_LIMBS] = t[FIELD::FIELD_LIMBS + 1] + carry;
  }

  FIELD result;
  for(unsigned i = 0; i < FIELD::FIELD_LIMBS; i++) result.val[i] = t[i];

  if(gte(result, *FIELD::Get_P_PTR())) result = FIELD::sub_(result, *FIELD::Get_P_PTR());

  return result;
}

/*
 * (a_0 + u * a_1)(b_0 + u * b_1) = a_0 * b_0 - a_1 * b_1 + u * (a_0 * b_1 + a_1 * b_0)
 * Therefore:
 * c_0 = a_0 * b_0 - a_1 * b_1
 * c_1 = (a_0 * b_1 + a_1 * b_0) = (a_0 + a_1) * (b_0 + b_1) - a_0 * b_0 - a_1 * b_1
 */
template<> cu_inline_func Fq2 mul(Fq2 a, Fq2 b) {
  const Fq aa = mul(a.c0, b.c0);
  const Fq bb = mul(a.c1, b.c1);
  const Fq o = add(b.c0, b.c1);
  a.c1 = add(a.c1, a.c0);
  a.c1 = mul(a.c1, o);
  a.c1 = sub(a.c1, aa);
  a.c1 = sub(a.c1, bb);
  a.c0 = sub(aa, bb);
  return a;
}

// Left-shift the limbs by one bit and subtract by modulus in case of overflow.
// Faster version of FIELD_add(a, a)
template<typename FIELD>
static cu_inline_func FIELD double_op(FIELD a) {
  for(unsigned i = FIELD::FIELD_LIMBS - 1; i >= 1; i--)
    a.val[i] = (a.val[i] << 1) | (a.val[i - 1] >> (FIELD::LIMB_BITS - 1));
  a.val[0] <<= 1;
  if(gte(a, *FIELD::Get_P_PTR())) a = FIELD::sub_(a, *FIELD::Get_P_PTR());
  return a;
}

template<> cu_inline_func Fq2 double_op(Fq2 a) {
  a.c0 = double_op(a.c0);
  a.c1 = double_op(a.c1);
  return a;
}

template<typename G>
static cu_func projective<G> double_op(projective<G> inp) {
  if(eq(inp.z, *G::Get_ZERO_PTR())) {
      return inp;
  }

  const G a = sqr(inp.x); // A = X1^2
  const G b = sqr(inp.y); // B = Y1^2
  G c = sqr(b); // C = B^2

  // D = 2*((X1+B)2-A-C)
  G d = add(inp.x, b);
  d = sqr(d); d = sub(sub(d, a), c); d = double_op(d);

  const G e = add(double_op(a), a); // E = 3*A
  const G f = sqr(e);

  inp.z = mul(inp.y, inp.z); inp.z = double_op(inp.z); // Z3 = 2*Y1*Z1
  inp.x = sub(sub(f, d), d); // X3 = F-2*D

  // Y3 = E*(D-X3)-8*C
  c = double_op(c); c = double_op(c); c = double_op(c);
  inp.y = sub(mul(sub(d, inp.x), e), c);

  return inp;
}

/*
 * (a_0 + u * a_1)(a_0 + u * a_1) = a_0 ^ 2 - a_1 ^ 2 + u * 2 * a_0 * a_1
 * Therefore:
 * c_0 = (a_0 * a_0 - a_1 * a_1) = (a_0 + a_1)(a_0 - a_1)
 * c_1 = 2 * a_0 * a_1
 */
template<> cu_inline_func Fq2 sqr(Fq2 a) {
  const Fq ab = mul(a.c0, a.c1);
  const Fq c0c1 = add(a.c0, a.c1);
  a.c0 = mul(sub(a.c0, a.c1), c0c1);
  a.c1 = double_op(ab);
  return a;
}

// Modular exponentiation (Exponentiation by Squaring)
// https://en.wikipedia.org/wiki/Exponentiation_by_squaring
template<typename FIELD>
static cu_inline_func FIELD pow(FIELD base, unsigned exponent) {
  FIELD res = FIELD::Get_ONE();
  while(exponent > 0) {
    if (exponent & 1)
      res = mul(res, base);
    exponent = exponent >> 1;
    base = sqr(base);
  }
  return res;
}


// Store squares of the base in a lookup table for faster evaluation.
template<typename FIELD>
static cu_inline_func FIELD pow_lookup(FIELD *bases, unsigned exponent) {
  FIELD res = FIELD::Get_ONE();
  unsigned i = 0;
  while(exponent > 0) {
    if (exponent & 1)
      res = mul(res, bases[i]);
    exponent = exponent >> 1;
    i++;
  }
  return res;
}

template<typename FIELD>
static cu_inline_func FIELD mont(FIELD a) {
  return mul(a, FIELD::R2);
}

template<typename FIELD>
static cu_inline_func FIELD unmont(FIELD a) {
  FIELD one = FIELD::ZERO;
  one.val[0] = 1;
  return mul(a, one);
}

// Get `i`th bit (From most significant digit) of the field.
template<typename FIELD>
static cu_inline_func bool get_bit(FIELD l, unsigned i) {
  return (l.val[FIELD::FIELD_LIMBS - 1 - i / FIELD::LIMB_BITS] >>
          (FIELD::LIMB_BITS - 1 - (i % FIELD::LIMB_BITS))) &1;
}

// Get `window` consecutive bits, (Starting from `skip`th bit) from the field.
template<typename FIELD>
static cu_inline_func unsigned get_bits(FIELD l, unsigned skip, unsigned window) {
  unsigned ret = 0;
  for(unsigned i = 0; i < window; i++) {
    ret <<= 1;
    ret |= get_bit(l, skip + i);
  }
  return ret;
}

template<typename FIELD>
static cu_func void print(FIELD a) {
  printf("0x");
  for (unsigned i = 0; i < FIELD::LIMBS; i++) {
    printf("%016lx", a.val[FIELD::LIMBS - i - 1]);
  }
}

#endif