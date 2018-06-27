import math
import json
import base64

import numpy as np

from numba import njit, int32


# this is a 1-to-1 translation of our js bloom filters to python


@njit(int32(int32))
def popcnt(v):
    v -= (v >> 1) & 0x55555555
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    return ((v + (v >> 4) & 0xf0f0f0f) * 0x1010101) >> 24


# a * 16777619 mod 2**32
@njit(int32(int32))
def fnv_multiply(a):
    return a + (a << 1) + (a << 4) + (a << 7) + (a << 8) + (a << 24)


#// See https://web.archive.org/web/20131019013225/http://home.comcast.net/~bretm/hash/6.html
@njit(int32(int32))
def fnv_mix(a):
    a += (a << 13)
    a ^= (a >> 7)
    a += (a << 3)
    a ^= (a >> 17)
    a += (a << 5)
    return a & 0xffffffff


#// Fowler/Noll/Vo hashing.
#// Nonstandard variation: this function optionally takes a seed value that is incorporated
#// into the offset basis. According to http://www.isthe.com/chongo/tech/comp/fnv/index.html
#// "almost any offset_basis will serve so long as it is non-zero".
@njit
def fnv_1a(v, seed):
    a = 2166136261 ^ seed

    for i in range(len(v)):
        c = v[i]
        d = c & 0xff00
        if d:
            a = fnv_multiply(a ^ d >> 8)
        a = fnv_multiply(a ^ c & 0xff)

    return fnv_mix(a)


@njit
def bf_calculate_locations(r, m, key):
    a = fnv_1a(key, 0)
    b = fnv_1a(key, 1576284489) # // The seed value is chosen randomly
    x = a % m

    for i in range(len(r)):
        if x < 0:
            r[i] = x + m
        else:
            r[i] = x 
        x = (x + b) % m

    return r


@njit
def bf_test(locations, buckets):
    for i in range(len(locations)):
        b = locations[i]

        if buckets[math.floor(b / 32)] & (1 << (b % 32)) == 0:
            return False

    return True


@njit
def bf_add(locations, buckets, key):
    for i in range(len(locations)):
        b = locations[i]
        buckets[math.floor(b / 32)] |= 1 << (b % 32)


@njit
def buckets_union(b1, b2):
    n1 = len(b1)
    n2 = len(b2)
    assert n1 == n2

    for i in range(n1):
        b1[i] = b1[i] | b2[i]

    return b1


def create_empty(capacity, error_rate=0.001):    
    if not (0 < error_rate < 1):
        raise ValueError("Error_Rate must be between 0 and 1.")
    if not capacity > 0:
        raise ValueError("Capacity must be > 0")

    num_bits = (-capacity * math.log(error_rate) / (math.log(2) * math.log(2)))
    num_hashes = max(1, round(num_bits / capacity * math.log(2)))

    n = math.ceil(num_bits / 32)
    buckets = np.zeros(n, dtype='int32')

    return BloomFilter(num_hashes, buckets)


class BloomFilter(object):
    def __init__(self, num_hashes, buckets):
        self.buckets = buckets
        self.num_hashes = num_hashes
        self.n = len(buckets)
        self.m = self.n * 32
        self._locations = np.zeros(self.num_hashes, dtype='uint32')

    def _calculate_locations(self, key):
        return bf_calculate_locations(self._locations, self.m, key)

    def _calculate_key(self, key):
        bkey = key.encode()
        #bkey = np.frombuffer(bkey, dtype='uint8')
        return bkey

    def test(self, key):
        key = self._calculate_key(key)
        l = self._calculate_locations(key)
        return bf_test(l, self.buckets)

    def __contains__(self, key):
        return self.test(key)

    def add(self, key):
        key = self._calculate_key(key)
        l = self._calculate_locations(key)
        bf_add(l, self.buckets, key)

    def union(self, other):
        assert self.num_hashes == other.num_hashes
        self.buckets = buckets_union(self.buckets, other.buckets)
        return self

    def save(self, file):
        save(self.num_hashes, self.buckets, file)


def save(num_hashes, buckets, file):
    buckets = np.array(buckets, dtype='int32')
    b64 = base64.b64encode(buckets.tobytes()).decode()

    d = dict(num_hashes=num_hashes, buckets=b64)

    with open(file, 'w') as f_out:
        json.dump(d, f_out)


def load(file):
    with open(file, 'r') as f_in:
        d = json.load(f_in)

    b64 = d['buckets']
    buckets = np.frombuffer(base64.b64decode(b64), dtype='int32')

    return BloomFilter(d['num_hashes'], buckets)