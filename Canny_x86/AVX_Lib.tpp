template <typename T>
void _mm512_memcpy(T* dest, const T* src, size_t size)
{
    // Calculate the number of elements to be copied per AVX512 register
    const size_t elements_per_register = 64 / sizeof(T);

    // Calculate the number of full AVX512 registers we can copy
    size_t num_full_registers = size / elements_per_register;

    // Copy data using AVX512 registers
    for (size_t i = 0; i < num_full_registers; ++i)
    {
        __m512i data = _mm512_load_si512(reinterpret_cast<const __m512i*>(src) + i);
        _mm512_store_si512(reinterpret_cast<__m512i*>(dest) + i, data);
    }

    // Calculate the number of remaining elements to be copied
    size_t remaining_elements = size % elements_per_register;
    size_t offset = num_full_registers * elements_per_register;

    // Copy remaining elements
    for (size_t i = 0; i < remaining_elements; ++i)
    {
        dest[offset + i] = src[offset + i];
    }
}