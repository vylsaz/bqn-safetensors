#include"bqnffi.h"

#include<windows.h>

void* _makeF(const char* name) {
    static HMODULE hModule = NULL;
    if (hModule==NULL) {
        hModule = GetModuleHandleA(NULL);
        if (hModule==NULL) {exit(1);}
    }
    return GetProcAddress(hModule, name);
}

#define INIT(F) \
    static typeof(F)* _f = NULL; \
    if (_f==NULL) { \
        _f = _makeF(#F); \
        if (_f==NULL) {exit(1);} \
    } 

void bqn_init(void) { INIT(bqn_init) _f(); }

void bqn_free(BQNV v) { INIT(bqn_free) _f(v); }
BQNV bqn_copy(BQNV v) { INIT(bqn_copy) return _f(v); }

double   bqn_toF64 (BQNV v) {
    INIT(bqn_toF64) return _f(v);
}
uint32_t bqn_toChar(BQNV v) {
    INIT(bqn_toChar) return _f(v);
}
double   bqn_readF64 (BQNV v) {
    INIT(bqn_readF64) return _f(v);
}
uint32_t bqn_readChar(BQNV v) {
    INIT(bqn_readChar) return _f(v);
}

int bqn_type(BQNV v) {
    INIT(bqn_type) return _f(v);
}

BQNV bqn_call1(BQNV f, BQNV x) {
    INIT(bqn_call1) return _f(f, x);
}
BQNV bqn_call2(BQNV f, BQNV w, BQNV x) {
    INIT(bqn_call2) return _f(f, w, x);
}

BQNV bqn_eval(BQNV src) {
    INIT(bqn_eval) return _f(src);
}
BQNV bqn_evalCStr(const char* str) {
    INIT(bqn_evalCStr) return _f(str);
}

size_t bqn_bound(BQNV a) {
    INIT(bqn_bound) return _f(a);
}
size_t bqn_rank(BQNV a) {
    INIT(bqn_rank) return _f(a);
}
void bqn_shape(BQNV a, size_t* buf) {
    INIT(bqn_shape) _f(a, buf);
} 
BQNV bqn_pick(BQNV a, size_t pos) {
    INIT(bqn_pick) return _f(a, pos);
} 

void bqn_readI8Arr (BQNV a, int8_t*   buf) {
    INIT(bqn_readI8Arr ) _f(a, buf);
}
void bqn_readI16Arr(BQNV a, int16_t*  buf) {
    INIT(bqn_readI16Arr) _f(a, buf);
}
void bqn_readI32Arr(BQNV a, int32_t*  buf) {
    INIT(bqn_readI32Arr) _f(a, buf);
}
void bqn_readF64Arr(BQNV a, double*   buf) {
    INIT(bqn_readF64Arr) _f(a, buf);
}
void bqn_readC8Arr (BQNV a, uint8_t*  buf) {
    INIT(bqn_readC8Arr ) _f(a, buf);
}
void bqn_readC16Arr(BQNV a, uint16_t* buf) {
    INIT(bqn_readC16Arr) _f(a, buf);
}
void bqn_readC32Arr(BQNV a, uint32_t* buf) {
    INIT(bqn_readC32Arr) _f(a, buf);
}
void bqn_readObjArr(BQNV a, BQNV*     buf) {
    INIT(bqn_readObjArr) _f(a, buf);
}

bool bqn_hasField(BQNV ns, BQNV name) {
    INIT(bqn_hasField) return _f(ns, name);
}
BQNV bqn_getField(BQNV ns, BQNV name) {
    INIT(bqn_getField) return _f(ns, name);
}

BQNV bqn_makeF64(double d) {
    INIT(bqn_makeF64) return _f(d);
}
BQNV bqn_makeChar(uint32_t c) {
    INIT(bqn_makeChar) return _f(c);
}

BQNV bqn_makeI8Arr (size_t rank, const size_t* shape, const int8_t*   data) {
    INIT(bqn_makeI8Arr ) return _f(rank, shape, data);
}
BQNV bqn_makeI16Arr(size_t rank, const size_t* shape, const int16_t*  data) {
    INIT(bqn_makeI16Arr) return _f(rank, shape, data);
}
BQNV bqn_makeI32Arr(size_t rank, const size_t* shape, const int32_t*  data) {
    INIT(bqn_makeI32Arr) return _f(rank, shape, data);
}
BQNV bqn_makeF64Arr(size_t rank, const size_t* shape, const double*   data) {
    INIT(bqn_makeF64Arr) return _f(rank, shape, data);
}
BQNV bqn_makeC8Arr (size_t rank, const size_t* shape, const uint8_t*  data) {
    INIT(bqn_makeC8Arr ) return _f(rank, shape, data);
}
BQNV bqn_makeC16Arr(size_t rank, const size_t* shape, const uint16_t* data) {
    INIT(bqn_makeC16Arr) return _f(rank, shape, data);
}
BQNV bqn_makeC32Arr(size_t rank, const size_t* shape, const uint32_t* data) {
    INIT(bqn_makeC32Arr) return _f(rank, shape, data);
}
BQNV bqn_makeObjArr(size_t rank, const size_t* shape, const BQNV*     data) {
    INIT(bqn_makeObjArr) return _f(rank, shape, data);
}

BQNV bqn_makeI8Vec (size_t len, const int8_t*   data) {
    INIT(bqn_makeI8Vec ) return _f(len, data);
}
BQNV bqn_makeI16Vec(size_t len, const int16_t*  data) {
    INIT(bqn_makeI16Vec) return _f(len, data);
}
BQNV bqn_makeI32Vec(size_t len, const int32_t*  data) {
    INIT(bqn_makeI32Vec) return _f(len, data);
}
BQNV bqn_makeF64Vec(size_t len, const double*   data) {
    INIT(bqn_makeF64Vec) return _f(len, data);
}
BQNV bqn_makeC8Vec (size_t len, const uint8_t*  data) {
    INIT(bqn_makeC8Vec ) return _f(len, data);
}
BQNV bqn_makeC16Vec(size_t len, const uint16_t* data) {
    INIT(bqn_makeC16Vec) return _f(len, data);
}
BQNV bqn_makeC32Vec(size_t len, const uint32_t* data) {
    INIT(bqn_makeC32Vec) return _f(len, data);
}
BQNV bqn_makeObjVec(size_t len, const BQNV*     data) {
    INIT(bqn_makeObjVec) return _f(len, data);
}
BQNV bqn_makeUTF8Str(size_t len, const char* str) {
    INIT(bqn_makeUTF8Str) return _f(len, str);
}

BQNV bqn_makeBoundFn1(bqn_boundFn1 f, BQNV obj) {
    INIT(bqn_makeBoundFn1) return _f(f, obj);
}
BQNV bqn_makeBoundFn2(bqn_boundFn2 f, BQNV obj) {
    INIT(bqn_makeBoundFn2) return _f(f, obj);
}

BQNElType bqn_directArrType(BQNV a) {
    INIT(bqn_directArrType) return _f(a);
}

const int8_t*   bqn_directI8 (BQNV a) {
    INIT(bqn_directI8 ) return _f(a);
}
const int16_t*  bqn_directI16(BQNV a) {
    INIT(bqn_directI16) return _f(a);
}
const int32_t*  bqn_directI32(BQNV a) {
    INIT(bqn_directI32) return _f(a);
}
const double*   bqn_directF64(BQNV a) {
    INIT(bqn_directF64) return _f(a);
}
const uint8_t*  bqn_directC8 (BQNV a) {
    INIT(bqn_directC8 ) return _f(a);
}
const uint16_t* bqn_directC16(BQNV a) {
    INIT(bqn_directC16) return _f(a);
}
const uint32_t* bqn_directC32(BQNV a) {
    INIT(bqn_directC32) return _f(a);
}
