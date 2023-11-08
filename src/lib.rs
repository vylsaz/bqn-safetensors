use cbqn_sys::{
    bqn_bound, bqn_directArrType, bqn_directF64, bqn_directI16, bqn_directI32, bqn_directI8,
    bqn_free, bqn_makeF64, bqn_makeF64Arr, bqn_makeI16Arr, bqn_makeI32Arr, bqn_makeI8Arr,
    bqn_makeObjVec, bqn_makeUTF8Str, bqn_pick, bqn_rank, bqn_readC32Arr, bqn_shape, bqn_type,
    BQNElType, BQNElType_elt_c16, BQNElType_elt_c32, BQNElType_elt_c8, BQNElType_elt_f64,
    BQNElType_elt_i16, BQNElType_elt_i32, BQNElType_elt_i8, BQNV,
};
use half::{bf16, f16};
use memmap2::{Mmap, MmapOptions};
use safetensors::tensor::{serialize_to_file, Dtype, SafeTensors, TensorView};
use std::{collections::HashMap, ffi::c_char, fs::File, path::PathBuf};
use widestring::U32String;

fn to_string(a: BQNV) -> Result<String, String> {
    unsafe {
        if bqn_type(a) != 0 {
            return Err(format!("Not a string"));
        }
        let bound = bqn_bound(a);
        let eltype: BQNElType = bqn_directArrType(a);
        if eltype != BQNElType_elt_c8 && eltype != BQNElType_elt_c16 && eltype != BQNElType_elt_c32
        {
            return Err(format!("Not a string"));
        }
        let mut s = Vec::with_capacity(bound);
        s.set_len(bound);
        bqn_readC32Arr(a, s.as_mut_ptr());
        Ok(U32String::from_vec(s).to_string_lossy())
    }
}

fn to_path_buf(a: BQNV) -> Result<PathBuf, String> {
    let name_str = to_string(a)?;
    Ok(PathBuf::from(name_str))
}

#[allow(non_upper_case_globals)]
fn serialize_impl(filepath: Result<PathBuf, String>, kv_array: BQNV) -> Result<(), String> {
    let filepath = filepath?;
    unsafe {
        let bound = bqn_bound(kv_array);
        if bound % 2 != 0 {
            return Err(format!("Length error"));
        }
        let half = bound / 2;
        let mut tensors = HashMap::with_capacity(half);
        for n in 0..half {
            let bqn_key = bqn_pick(kv_array, 2 * n);
            let bqn_val = bqn_pick(kv_array, 2 * n + 1);

            let key = to_string(bqn_key);
            let val = if bqn_type(bqn_val) != 0 {
                Err(format!("Item at index {n} is not an array"))
            } else {
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
                        TensorView::new(Dtype::I8, shape, data)
                            .map_err(|e| format!("SafeTensors: {e}"))
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
            };

            bqn_free(bqn_key);
            bqn_free(bqn_val);
            tensors.insert(key?, val?);
        }
        serialize_to_file(&tensors, &None, &filepath).map_err(|e| format!("SafeTensors: {e}"))
    }
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

fn get_names_impl(filepath: Result<PathBuf, String>) -> Result<BQNV, String> {
    let filepath = filepath?;
    let file = File::open(filepath).map_err(|e| format!("{e}"))?;
    let buffer: Mmap = unsafe { MmapOptions::new().map(&file).map_err(|e| format!("{e}"))? };
    let (_, metadata) =
        SafeTensors::read_metadata(&buffer).map_err(|e| format!("SafeTensors: {e:?}"))?;
    let mut keys: Vec<String> = metadata.tensors().keys().cloned().collect();
    keys.sort();
    let bqn_keys: Vec<BQNV> = keys
        .into_iter()
        .map(|k| unsafe { bqn_makeUTF8Str(k.len(), k.as_ptr() as *const c_char) })
        .collect();
    let ret = unsafe { bqn_makeObjVec(bqn_keys.len(), bqn_keys.as_ptr()) };
    Ok(ret)
}

fn get_tensor_impl(
    filepath: Result<PathBuf, String>,
    name: Result<String, String>,
) -> Result<BQNV, String> {
    let filepath = filepath?;
    let name = name?;
    let file = File::open(filepath).map_err(|e| format!("{e}"))?;
    let buffer: Mmap = unsafe { MmapOptions::new().map(&file).map_err(|e| format!("{e}"))? };
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
    let ret = unsafe {
        match info.dtype {
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
        }
    };
    Ok(ret)
}

fn make_return(result: Result<BQNV, String>) -> BQNV {
    unsafe {
        result
            .map(|o| bqn_makeObjVec(2, [bqn_makeF64(1.0), o].as_ptr()))
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
}

#[no_mangle]
pub extern "C" fn serialize(filename: BQNV, kv_array: BQNV) -> BQNV {
    unsafe {
        let ret =
            make_return(serialize_impl(to_path_buf(filename), kv_array).map(|_| bqn_makeF64(1.0)));
        bqn_free(filename);
        bqn_free(kv_array);
        ret
    }
}

#[no_mangle]
pub extern "C" fn get_names(filename: BQNV) -> BQNV {
    unsafe {
        let ret = make_return(get_names_impl(to_path_buf(filename)));
        bqn_free(filename);
        ret
    }
}

#[no_mangle]
pub extern "C" fn get_tensor(filename: BQNV, tensor_k: BQNV) -> BQNV {
    unsafe {
        let ret = make_return(get_tensor_impl(to_path_buf(filename), to_string(tensor_k)));
        bqn_free(filename);
        bqn_free(tensor_k);
        ret
    }
}
