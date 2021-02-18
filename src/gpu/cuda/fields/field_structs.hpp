#ifndef FIELD_STRUCTS_HEADER
#define FIELD_STRUCTS_HEADER

#include <stdint.h>

#define FR_BITS 256
#define FQ_BITS 384


typedef uint64_t limb_type;
#define GLOBAL_LIMB_BITS 64

// 256 bit big integers
typedef struct Fr {
    limb_type val[FR_BITS / GLOBAL_LIMB_BITS];
#ifdef __CUDACC__
    static constexpr int FIELD_LIMBS = 4;
    static constexpr int LIMB_BITS = 64;

    typedef ::limb_type limb_type;
    cu_inline_func static limb_type add_with_carry(limb_type a, limb_type *b);
    cu_inline_func static limb_type mac_with_carry(limb_type a, limb_type b, limb_type c, limb_type *d);
    cu_inline_func static Fr add_(Fr a, Fr b);
    cu_inline_func static Fr sub_(Fr a, Fr b);
    cu_inline_func static Fr Get_ONE();
    cu_inline_func static Fr Get_P();
    cu_inline_func static Fr Get_ZERO();
    cu_inline_func static Fr Get_R2();
    cu_inline_func static Fr *Get_ONE_PTR();
    cu_inline_func static Fr *Get_P_PTR();
    cu_inline_func static Fr *Get_ZERO_PTR();
    cu_inline_func static Fr *Get_R2_PTR();
    static const Fr P;
    static const Fr ONE;
    static const Fr ZERO;
    static const Fr R2;
    static const limb_type INV;
#endif
} Fr;

  
// 384 bit big integer 
typedef struct Fq {
  limb_type val[FQ_BITS / GLOBAL_LIMB_BITS];
#ifdef __CUDACC__
  static constexpr int FIELD_LIMBS = 6;
  static constexpr int LIMB_BITS = 64;

  typedef ::limb_type limb_type;
  cu_inline_func static limb_type add_with_carry(limb_type a, limb_type *b);
  cu_inline_func static limb_type mac_with_carry(limb_type a, limb_type b, limb_type c, limb_type *d);
  cu_inline_func static Fq add_(Fq a, Fq b);
  cu_inline_func static Fq sub_(Fq a, Fq b);
  cu_inline_func static Fq Get_ONE();
  cu_inline_func static Fq *Get_ONE_PTR();
  cu_inline_func static Fq Get_P();
  cu_inline_func static Fq *Get_P_PTR();
  cu_inline_func static Fq Get_ZERO();
  cu_inline_func static Fq *Get_ZERO_PTR();
  cu_inline_func static Fq Get_R2();
  cu_inline_func static Fq *Get_R2_PTR();
  static const Fq P;
  static const Fq ONE;
  static const Fq ZERO;
  static const Fq R2;
  static const limb_type INV;
#endif
} Fq;

typedef struct Fq2 {
    Fq c0;
    Fq c1;
#ifdef __CUDACC__
    typedef Fq2 field_type;
    typedef ::limb_type limb_type;
    static const int LIMB_BITS;
    static const Fq2 ZERO;
    static const Fq2 ONE;
    cu_inline_func static Fq2 Get_ONE();
    cu_inline_func static Fq2 *Get_ONE_PTR();
    cu_inline_func static Fq2 Get_ZERO();
    cu_inline_func static Fq2 *Get_ZERO_PTR();
#endif
} Fq2; // Represents: c0 + u * c1

template<typename T>
struct affine {
    T x;
    T y;
#ifndef BLSTRS
    bool inf;
#endif
#ifndef USE_64BIT // 32bit needs padding
// uint32_t _padding;
#endif
};

template<typename T>
struct projective {
    T x;
    T y;
    T z;
#ifdef __CUDACC__
    cu_inline_func static projective<T> Get_ZERO();
    cu_inline_func static projective<T> *Get_ZERO_PTR();
    static const projective<T> ZERO;
    cu_inline_func static void COPY_ZERO(projective<T> *d);
#endif
};

#endif
