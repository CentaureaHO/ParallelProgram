#pragma once
#include <immintrin.h>

__m512 _mm512_arctan2(__m512 y, __m512 x);

__m128i cvtepi32_epi8(__m512i v);

template <typename T>
void _mm512_memcpy(T* dest, const T* src, size_t size);

#include "AVX_Lib.tpp"