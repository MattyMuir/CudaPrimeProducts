#pragma once
#include <stdint.h>

struct Divider
{
    uint64_t d, magic;
    uint8_t more;
};

__device__ uint64_t Div128(uint64_t numhi, uint64_t den, uint64_t* r)
{
    constexpr uint64_t b = ((uint64_t)1 << 32);

    int shift = __clzll(den);
    //int shift = 63 - Log2(den);
    den <<= shift;
    numhi <<= shift;

    uint32_t den1 = (uint32_t)(den >> 32);
    uint32_t den0 = (uint32_t)(den & 0xFFFFFFFFu);

    uint64_t qhat = numhi / den1;
    uint64_t rhat = numhi % den1;
    uint64_t c1 = qhat * den0;
    uint64_t c2 = rhat * b;
    if (c1 > c2) { qhat -= (c1 - c2 > den) ? 2 : 1; }
    uint32_t q1 = (uint32_t)qhat;

    uint64_t rem = numhi * b - q1 * den;

    qhat = rem / den1;
    rhat = rem % den1;
    c1 = qhat * den0;
    c2 = rhat * b;
    if (c1 > c2) { qhat -= (c1 - c2 > den) ? 2 : 1; }
    uint32_t q0 = (uint32_t)qhat;

    *r = (rem * b - q0 * den) >> shift;
    return ((uint64_t)q1 << 32) | q0;
}

__device__ Divider GenDiv(uint64_t d)
{
    Divider res;
    res.d = d;
    uint32_t dLog = 63 - __clzll(d);
    //uint32_t dLog = Log2(d);

    uint64_t rem;
    uint64_t m = 2 * Div128((uint64_t)1 << dLog, d, &rem);

    const uint64_t twice_rem = rem + rem;
    if (twice_rem >= d || twice_rem < rem) { m += 1; };
    uint8_t more = (uint8_t)(dLog | 0x40);
    res.magic = 1 + m;
    res.more = more;

    return res;
}

__device__ uint64_t Divide(uint64_t x, const Divider& d)
{
    uint8_t more = d.more;
    uint64_t q = __umul64hi(d.magic, x);
    return  (((x - q) >> 1) + q) >> (more & 0x3F);
}