//! Little-endian binary (de)serialization primitives for on-disk persistence.
//!
//! Deliberately hand-rolled rather than pulling in `serde` + `bincode`: the
//! format is small, the code is readable, and the crate stays dependency-free.
//! Everything is fixed-width little-endian; strings and payloads are
//! length-prefixed.

use std::io::{self, Read, Write};

use crate::payload::{Payload, Value};

pub(crate) fn write_u8<W: Write>(w: &mut W, v: u8) -> io::Result<()> {
    w.write_all(&[v])
}

pub(crate) fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

pub(crate) fn write_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

pub(crate) fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

pub(crate) fn write_u64<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

pub(crate) fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

pub(crate) fn write_i64<W: Write>(w: &mut W, v: i64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

pub(crate) fn read_i64<R: Read>(r: &mut R) -> io::Result<i64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(i64::from_le_bytes(b))
}

pub(crate) fn write_f32<W: Write>(w: &mut W, v: f32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

pub(crate) fn read_f32<R: Read>(r: &mut R) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

pub(crate) fn write_f64<W: Write>(w: &mut W, v: f64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

pub(crate) fn read_f64<R: Read>(r: &mut R) -> io::Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}

pub(crate) fn write_string<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    write_u32(w, s.len() as u32)?;
    w.write_all(s.as_bytes())
}

pub(crate) fn read_string<R: Read>(r: &mut R) -> io::Result<String> {
    let len = read_u32(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

pub(crate) fn write_payload<W: Write>(w: &mut W, payload: Option<&Payload>) -> io::Result<()> {
    match payload {
        None => write_u8(w, 0),
        Some(p) => {
            write_u8(w, 1)?;
            write_u32(w, p.len() as u32)?;
            for (key, value) in p {
                write_string(w, key)?;
                write_value(w, value)?;
            }
            Ok(())
        }
    }
}

pub(crate) fn read_payload<R: Read>(r: &mut R) -> io::Result<Option<Payload>> {
    if read_u8(r)? == 0 {
        return Ok(None);
    }
    let fields = read_u32(r)? as usize;
    let mut payload = Payload::new();
    for _ in 0..fields {
        let key = read_string(r)?;
        let value = read_value(r)?;
        payload.insert(key, value);
    }
    Ok(Some(payload))
}

fn write_value<W: Write>(w: &mut W, value: &Value) -> io::Result<()> {
    match value {
        Value::Str(s) => {
            write_u8(w, 0)?;
            write_string(w, s)
        }
        Value::Int(i) => {
            write_u8(w, 1)?;
            write_i64(w, *i)
        }
        Value::Float(f) => {
            write_u8(w, 2)?;
            write_f64(w, *f)
        }
        Value::Bool(b) => {
            write_u8(w, 3)?;
            write_u8(w, u8::from(*b))
        }
    }
}

fn read_value<R: Read>(r: &mut R) -> io::Result<Value> {
    Ok(match read_u8(r)? {
        0 => Value::Str(read_string(r)?),
        1 => Value::Int(read_i64(r)?),
        2 => Value::Float(read_f64(r)?),
        3 => Value::Bool(read_u8(r)? != 0),
        tag => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown payload value tag {tag}"),
            ))
        }
    })
}
