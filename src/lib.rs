use cbqn_sys::{
    bqn_bound, bqn_directArrType, bqn_directF64, bqn_directI16, bqn_directI32, bqn_directI8,
    bqn_free, bqn_makeChar, bqn_makeF64, bqn_makeF64Arr, bqn_makeI16Arr, bqn_makeI32Arr,
    bqn_makeI8Arr, bqn_makeObjVec, bqn_makeUTF8Str, bqn_pick, bqn_rank, bqn_readC8Arr, bqn_shape,
    BQNElType, BQNElType_elt_f64, BQNElType_elt_i16, BQNElType_elt_i32, BQNElType_elt_i8, BQNV,
};
use half::{bf16, f16};
use memmap2::{Mmap, MmapOptions};
use safetensors::tensor::{serialize_to_file, Dtype, SafeTensors, TensorView};
use std::{collections::HashMap, ffi::c_char, fs::File, path::PathBuf};

unsafe fn get_string(a: BQNV) -> String {
    let bound = bqn_bound(a);
    let mut s = Vec::with_capacity(bound);
    s.set_len(bound);
    bqn_readC8Arr(a, s.as_mut_ptr());
    String::from_utf8_unchecked(s)
}

#[allow(non_upper_case_globals)]
unsafe fn serialize_impl(filename: BQNV, k_array: BQNV, v_array: BQNV) -> Result<BQNV, String> {
    let filename = PathBuf::from(get_string(filename));
    let bound = bqn_bound(k_array);
    let mut tensors = HashMap::with_capacity(bound);
    for i in 0..bound {
        let bqn_key = bqn_pick(k_array, i);
        let bqn_val = bqn_pick(v_array, i);

        let key = get_string(bqn_key);
        let val = {
            let len = bqn_bound(bqn_val);
            let rank = bqn_rank(bqn_val);
            let mut shape = Vec::with_capacity(rank);
            shape.set_len(rank);
            bqn_shape(bqn_val, shape.as_mut_ptr());
            let eltype: BQNElType = bqn_directArrType(bqn_val);
            match eltype {
                // TODO: endianness awareness, assumed LE for now
                BQNElType_elt_i8 => {
                    let ptr = bqn_directI8(bqn_val) as *const u8;
                    let data = std::slice::from_raw_parts(ptr, 1 * len);
                    TensorView::new(Dtype::I8, shape, data).map_err(|e| format!("SafeTensors: {e}"))
                }
                BQNElType_elt_i16 => {
                    let ptr = bqn_directI16(bqn_val) as *const u8;
                    let data = std::slice::from_raw_parts(ptr, 2 * len);
                    TensorView::new(Dtype::I16, shape, data)
                        .map_err(|e| format!("SafeTensors: {e}"))
                }
                BQNElType_elt_i32 => {
                    let ptr = bqn_directI32(bqn_val) as *const u8;
                    let data = std::slice::from_raw_parts(ptr, 4 * len);
                    TensorView::new(Dtype::I32, shape, data)
                        .map_err(|e| format!("SafeTensors: {e}"))
                }
                BQNElType_elt_f64 => {
                    let ptr = bqn_directF64(bqn_val) as *const u8;
                    let data = std::slice::from_raw_parts(ptr, 8 * len);
                    TensorView::new(Dtype::F64, shape, data)
                        .map_err(|e| format!("SafeTensors: {e}"))
                }
                _ => Err(format!("Element type not supported")),
            }
        }?;

        bqn_free(bqn_key);
        bqn_free(bqn_val);
        tensors.insert(key, val);
    }
    serialize_to_file(&tensors, &None, &filename)
        .map(|_| bqn_makeChar(0))
        .map_err(|e| format!("SafeTensors: {e}"))
}

unsafe fn get_names_impl(filename: BQNV) -> Result<BQNV, String> {
    let path = PathBuf::from(get_string(filename));
    let file = File::open(path).map_err(|e| format!("{e}"))?;
    let buffer: Mmap = MmapOptions::new().map(&file).map_err(|e| format!("{e}"))?;
    let (_, metadata) =
        SafeTensors::read_metadata(&buffer).map_err(|e| format!("SafeTensors: {e:?}"))?;
    let mut keys: Vec<String> = metadata.tensors().keys().cloned().collect();
    keys.sort();
    let bqn_keys: Vec<BQNV> = keys
        .into_iter()
        .map(|k| bqn_makeUTF8Str(k.len(), k.as_ptr() as *const c_char))
        .collect();
    let ret = bqn_makeObjVec(bqn_keys.len(), bqn_keys.as_ptr());
    Ok(ret)
}

macro_rules! convert_for_type {
    ($t1:ty, $t2:ty, $tensor:ident, $fn:ident, $rnk:ident, $shp:ident) => {{
        let it = $tensor.chunks(std::mem::size_of::<$t1>());
        let mut v: Vec<$t2> = Vec::with_capacity(it.len());
        for c in it {
            let r = c[0..std::mem::size_of::<$t1>()]
                .try_into()
                .map_err(|e| format!("{e}"))?;
            let u = <$t1>::from_le_bytes(r);
            v.push(<$t2>::from(u));
        }
        $fn($rnk, $shp, v.as_ptr())
    }};
}

macro_rules! convert_for_type_lossy {
    ($t1:ty, $t2:ty, $tensor:ident, $fn:ident, $rnk:ident, $shp:ident) => {{
        let it = $tensor.chunks(std::mem::size_of::<$t1>());
        let mut v: Vec<$t2> = Vec::with_capacity(it.len());
        for c in it {
            let r = c[0..std::mem::size_of::<$t1>()]
                .try_into()
                .map_err(|e| format!("{e}"))?;
            let u = <$t1>::from_le_bytes(r);
            v.push(u as $t2);
        }
        $fn($rnk, $shp, v.as_ptr())
    }};
}

unsafe fn get_tensor_impl(filename: BQNV, name: BQNV) -> Result<BQNV, String> {
    let path = PathBuf::from(get_string(filename));
    let name = get_string(name);
    let file = File::open(path).map_err(|e| format!("{e}"))?;
    let buffer: Mmap = MmapOptions::new().map(&file).map_err(|e| format!("{e}"))?;
    let (n, metadata) =
        SafeTensors::read_metadata(&buffer).map_err(|e| format!("SafeTensors: {e:?}"))?;
    let info = metadata
        .info(&name)
        .ok_or_else(|| format!("SafeTensors: File does not contain tensor {name}"))?;
    let offset = n + 8;
    let tensor = &buffer[info.data_offsets.0 + offset..info.data_offsets.1 + offset];
    let dat = tensor.as_ptr();
    let rnk = info.shape.len();
    let shp = info.shape.as_ptr();
    let ret = match info.dtype {
        Dtype::BOOL => bqn_makeI8Arr(rnk, shp, dat as *const i8),
        Dtype::I8 => bqn_makeI8Arr(rnk, shp, dat as *const i8),
        Dtype::I16 => bqn_makeI16Arr(rnk, shp, dat as *const i16),
        Dtype::I32 => bqn_makeI32Arr(rnk, shp, dat as *const i32),
        Dtype::F64 => bqn_makeF64Arr(rnk, shp, dat as *const f64),

        Dtype::U8 => {
            let v: Vec<i16> = tensor.iter().map(|c| i16::from(*c)).collect();
            bqn_makeI16Arr(rnk, shp, v.as_ptr())
        }
        Dtype::U16 => convert_for_type!(u16, i32, tensor, bqn_makeI32Arr, rnk, shp),
        Dtype::U32 => convert_for_type!(u32, f64, tensor, bqn_makeF64Arr, rnk, shp),
        Dtype::I64 => convert_for_type_lossy!(i64, f64, tensor, bqn_makeF64Arr, rnk, shp),
        Dtype::U64 => convert_for_type_lossy!(u64, f64, tensor, bqn_makeF64Arr, rnk, shp),
        Dtype::F32 => convert_for_type!(f32, f64, tensor, bqn_makeF64Arr, rnk, shp),
        Dtype::F16 => convert_for_type!(f16, f64, tensor, bqn_makeF64Arr, rnk, shp),
        Dtype::BF16 => convert_for_type!(bf16, f64, tensor, bqn_makeF64Arr, rnk, shp),
        _ => return Err(format!("Element type not supported")),
    };
    Ok(ret)
}

unsafe fn make_return(ret: Result<BQNV, String>) -> BQNV {
    ret.map(|o| bqn_makeObjVec(2, [bqn_makeF64(1.0), o].as_ptr()))
        .unwrap_or_else(|e| {
            bqn_makeObjVec(
                2,
                [
                    bqn_makeF64(0.0),
                    bqn_makeUTF8Str(e.len(), e.as_ptr() as *const c_char),
                ]
                .as_ptr(),
            )
        })
}

#[no_mangle]
pub unsafe extern "C" fn serialize(filename: BQNV, k_array: BQNV, v_array: BQNV) -> BQNV {
    let ret = serialize_impl(filename, k_array, v_array);
    bqn_free(filename);
    bqn_free(k_array);
    bqn_free(v_array);
    make_return(ret)
}

#[no_mangle]
pub unsafe extern "C" fn get_names(filename: BQNV) -> BQNV {
    let ret = get_names_impl(filename);
    bqn_free(filename);
    make_return(ret)
}

#[no_mangle]
pub unsafe extern "C" fn get_tensor(filename: BQNV, tensor_k: BQNV) -> BQNV {
    let ret = get_tensor_impl(filename, tensor_k);
    bqn_free(filename);
    bqn_free(tensor_k);
    make_return(ret)
}
