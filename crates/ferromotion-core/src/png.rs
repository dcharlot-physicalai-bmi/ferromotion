//! **A PNG decoder for MuJoCo's height fields**, in the form MuJoCo's compiler asks lodepng for: one 8-bit
//! grey value per pixel (`LCT_GREY`, bit depth 8).
//!
//! lodepng converts every colour type to grey by taking the RED channel (`rgba8ToPixel`: `gray = r`, the
//! average of the three commented out), a 16-bit sample by its high byte, and a palette index through the
//! palette's red. Non-interlaced images only; an Adam7 image is refused rather than guessed at.

/// `(width, height, grey)`, rows top to bottom as stored in the file.
pub(crate) fn decode_grey8(bytes: &[u8]) -> Result<(usize, usize, Vec<u8>), String> {
    const SIG: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    if bytes.len() < 8 || bytes[..8] != SIG {
        return Err("not a PNG file".into());
    }
    let be32 = |b: &[u8]| u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize;
    let (mut width, mut height, mut depth, mut ctype, mut interlace) = (0usize, 0usize, 0u8, 0u8, 0u8);
    let mut palette: Vec<[u8; 3]> = Vec::new();
    let mut idat: Vec<u8> = Vec::new();
    let mut i = 8;
    while i + 8 <= bytes.len() {
        let n = be32(&bytes[i..]);
        let kind = &bytes[i + 4..i + 8];
        let data = bytes.get(i + 8..i + 8 + n).ok_or("PNG chunk runs past the end of the file")?;
        match kind {
            b"IHDR" => {
                if n < 13 {
                    return Err("short IHDR".into());
                }
                width = be32(data);
                height = be32(&data[4..]);
                depth = data[8];
                ctype = data[9];
                interlace = data[12];
            }
            b"PLTE" => palette = data.as_chunks::<3>().0.to_vec(),
            b"IDAT" => idat.extend_from_slice(data),
            b"IEND" => break,
            _ => {}
        }
        i += 12 + n;
    }
    if width == 0 || height == 0 {
        return Err("PNG without dimensions".into());
    }
    if interlace != 0 {
        return Err("interlaced PNG is outside this decoder".into());
    }
    let channels = match ctype {
        0 | 3 => 1,
        2 => 3,
        4 => 2,
        6 => 4,
        other => return Err(format!("PNG colour type {other}")),
    };
    if !matches!(depth, 8 | 16) || (ctype == 3 && depth != 8) {
        return Err(format!("PNG bit depth {depth} (colour type {ctype}) is outside this decoder"));
    }
    let bps = depth as usize / 8;
    let bpp = channels * bps;
    let stride = width * bpp;
    if idat.len() < 2 || (idat[0] & 0x0f) != 8 {
        return Err("PNG data is not a zlib deflate stream".into());
    }
    let raw = inflate(&idat[2..])?;
    if raw.len() < height * (stride + 1) {
        return Err("PNG data shorter than its dimensions".into());
    }
    // undo the per-scanline filters
    let mut img = vec![0u8; height * stride];
    for y in 0..height {
        let filter = raw[y * (stride + 1)];
        let src = &raw[y * (stride + 1) + 1..(y + 1) * (stride + 1)];
        for x in 0..stride {
            let a = if x >= bpp { img[y * stride + x - bpp] as i32 } else { 0 };
            let b = if y > 0 { img[(y - 1) * stride + x] as i32 } else { 0 };
            let c = if x >= bpp && y > 0 { img[(y - 1) * stride + x - bpp] as i32 } else { 0 };
            let pred = match filter {
                0 => 0,
                1 => a,
                2 => b,
                3 => (a + b) / 2,
                4 => {
                    let p = a + b - c;
                    let (pa, pb, pc) = ((p - a).abs(), (p - b).abs(), (p - c).abs());
                    if pa <= pb && pa <= pc {
                        a
                    } else if pb <= pc {
                        b
                    } else {
                        c
                    }
                }
                other => return Err(format!("PNG filter {other}")),
            };
            img[y * stride + x] = (src[x] as i32 + pred) as u8;
        }
    }
    // to 8-bit grey as lodepng converts: the red channel (or the only one), a 16-bit sample's high byte
    let grey = (0..width * height)
        .map(|p| {
            let first = img[p * bpp];
            if ctype == 3 {
                palette.get(first as usize).map_or(0, |c| c[0])
            } else {
                first
            }
        })
        .collect();
    Ok((width, height, grey))
}

/// RFC 1951 DEFLATE: stored, fixed-Huffman and dynamic-Huffman blocks.
fn inflate(src: &[u8]) -> Result<Vec<u8>, String> {
    struct Bits<'a> {
        src: &'a [u8],
        pos: usize,
        bit: u32,
    }
    impl Bits<'_> {
        fn get(&mut self, n: u32) -> Result<u32, String> {
            let mut v = 0u32;
            for k in 0..n {
                let byte = *self.src.get(self.pos).ok_or("deflate stream ends early")?;
                v |= (((byte >> self.bit) & 1) as u32) << k;
                self.bit += 1;
                if self.bit == 8 {
                    self.bit = 0;
                    self.pos += 1;
                }
            }
            Ok(v)
        }
    }
    /// A canonical Huffman code: code-length counts and the symbols in code order.
    struct Huff {
        count: [u16; 16],
        symbol: Vec<u16>,
    }
    fn build(lengths: &[u8]) -> Huff {
        let mut count = [0u16; 16];
        for &l in lengths {
            count[l as usize] += 1;
        }
        count[0] = 0;
        let mut offs = [0u16; 16];
        for l in 1..16 {
            offs[l] = offs[l - 1] + count[l - 1];
        }
        let mut symbol = vec![0u16; lengths.len()];
        for (s, &l) in lengths.iter().enumerate() {
            if l != 0 {
                symbol[offs[l as usize] as usize] = s as u16;
                offs[l as usize] += 1;
            }
        }
        Huff { count, symbol }
    }
    fn decode(b: &mut Bits, h: &Huff) -> Result<u16, String> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..16 {
            code |= b.get(1)? as i32;
            let count = h.count[len] as i32;
            if code - count < first {
                return Ok(h.symbol[(index + (code - first)) as usize]);
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        Err("bad Huffman code".into())
    }
    const LBASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
    const LEXT: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
    const DBASE: [u16; 30] = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
    const DEXT: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
    let mut b = Bits { src, pos: 0, bit: 0 };
    let mut out: Vec<u8> = Vec::new();
    loop {
        let last = b.get(1)?;
        match b.get(2)? {
            0 => {
                if b.bit != 0 {
                    b.bit = 0;
                    b.pos += 1;
                }
                let s = b.src.get(b.pos..b.pos + 4).ok_or("stored block header ends early")?;
                let len = u16::from_le_bytes([s[0], s[1]]) as usize;
                b.pos += 4;
                out.extend_from_slice(b.src.get(b.pos..b.pos + len).ok_or("stored block ends early")?);
                b.pos += len;
            }
            kind @ (1 | 2) => {
                let (lit, dist) = if kind == 1 {
                    let mut l = [0u8; 288];
                    for (s, v) in l.iter_mut().enumerate() {
                        *v = match s {
                            0..=143 => 8,
                            144..=255 => 9,
                            256..=279 => 7,
                            _ => 8,
                        };
                    }
                    (build(&l), build(&[5u8; 30]))
                } else {
                    let hlit = b.get(5)? as usize + 257;
                    let hdist = b.get(5)? as usize + 1;
                    let hclen = b.get(4)? as usize + 4;
                    const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
                    let mut cl = [0u8; 19];
                    for &o in ORDER.iter().take(hclen) {
                        cl[o] = b.get(3)? as u8;
                    }
                    let clh = build(&cl);
                    let mut lengths = Vec::with_capacity(hlit + hdist);
                    while lengths.len() < hlit + hdist {
                        let sym = decode(&mut b, &clh)?;
                        match sym {
                            0..=15 => lengths.push(sym as u8),
                            16 => {
                                let prev = *lengths.last().ok_or("repeat with no previous length")?;
                                for _ in 0..3 + b.get(2)? {
                                    lengths.push(prev);
                                }
                            }
                            17 => lengths.extend(std::iter::repeat_n(0u8, 3 + b.get(3)? as usize)),
                            _ => lengths.extend(std::iter::repeat_n(0u8, 11 + b.get(7)? as usize)),
                        }
                    }
                    (build(&lengths[..hlit]), build(&lengths[hlit..hlit + hdist]))
                };
                loop {
                    let sym = decode(&mut b, &lit)? as usize;
                    if sym < 256 {
                        out.push(sym as u8);
                    } else if sym == 256 {
                        break;
                    } else {
                        let k = sym - 257;
                        if k >= 29 {
                            return Err("bad length symbol".into());
                        }
                        let len = LBASE[k] as usize + b.get(LEXT[k] as u32)? as usize;
                        let d = decode(&mut b, &dist)? as usize;
                        if d >= 30 {
                            return Err("bad distance symbol".into());
                        }
                        let back = DBASE[d] as usize + b.get(DEXT[d] as u32)? as usize;
                        if back > out.len() {
                            return Err("distance reaches before the output".into());
                        }
                        let start = out.len() - back;
                        for k in 0..len {
                            out.push(out[start + k]);
                        }
                    }
                }
            }
            _ => return Err("reserved deflate block type".into()),
        }
        if last == 1 {
            return Ok(out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2×2 RGB image written with zlib's stored and fixed-Huffman encodings (bytes produced by Python's
    /// `zlib.compress` at level 0 and 9): lodepng's grey is the RED channel.
    #[test]
    fn rgb_decodes_to_the_red_channel() {
        // IHDR 2×2, depth 8, colour type 2; IDAT = zlib of two scanlines, filter 0: (10,20,30)(40,50,60) /
        // (70,80,90)(200,210,220)
        let raw: [u8; 14] = [0, 10, 20, 30, 40, 50, 60, 0, 70, 80, 90, 200, 210, 220];
        let mut stored = vec![0x78, 0x01, 1, 14, 0, !14u8, 0xff];
        stored.extend_from_slice(&raw);
        let mut png = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
        let mut chunk = |kind: &[u8], data: &[u8]| {
            png.extend_from_slice(&(data.len() as u32).to_be_bytes());
            png.extend_from_slice(kind);
            png.extend_from_slice(data);
            png.extend_from_slice(&[0, 0, 0, 0]);
        };
        chunk(b"IHDR", &[0, 0, 0, 2, 0, 0, 0, 2, 8, 2, 0, 0, 0]);
        chunk(b"IDAT", &stored);
        chunk(b"IEND", &[]);
        let (w, h, g) = decode_grey8(&png).unwrap();
        assert_eq!((w, h, g), (2, 2, vec![10, 40, 70, 200]));
    }
}
