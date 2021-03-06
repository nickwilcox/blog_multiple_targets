#![allow(dead_code)]

use std::default::Default;

// In the following code we use the following acronyms for conciseness
// FL - front left
// FR - front right
// FC - front center, also just called center
// LF - low frequency, also called low frequency effects, or sub-woofer
// SL - surround left
// SR - surround right
// BL - back left
// BR - back right

#[derive(Default, Copy, Clone, Eq, PartialEq, Debug)]
pub struct InterleavedSample71 {
    fl: i16,
    fr: i16,
    fc: i16,
    lf: i16,
    sl: i16,
    sr: i16,
    bl: i16,
    br: i16,
}

pub struct InterleavedBuffer71 {
    data: Vec<InterleavedSample71>,
}

impl InterleavedBuffer71 {
    pub fn empty(num_samples: usize) -> Self {
        Self {
            data: vec![Default::default(); num_samples],
        }
    }
}

pub struct DeinterleavedBuffer71 {
    num_samples: usize,
    data_fl: Vec<f32>,
    data_fr: Vec<f32>,
    data_fc: Vec<f32>,
    data_lf: Vec<f32>,
    data_sl: Vec<f32>,
    data_sr: Vec<f32>,
    data_bl: Vec<f32>,
    data_br: Vec<f32>,
}

impl DeinterleavedBuffer71 {
    pub fn empty(num_samples: usize) -> Self {
        Self {
            num_samples,
            data_fl: vec![Default::default(); num_samples],
            data_fr: vec![Default::default(); num_samples],
            data_fc: vec![Default::default(); num_samples],
            data_lf: vec![Default::default(); num_samples],
            data_sl: vec![Default::default(); num_samples],
            data_sr: vec![Default::default(); num_samples],
            data_bl: vec![Default::default(); num_samples],
            data_br: vec![Default::default(); num_samples],
        }
    }
}

static POS_FLOAT_TO_16_SCALE: f32 = 0x7fff as f32;

#[inline(always)]
fn pcm_float_to_16(x: f32) -> i16 {
    (x * POS_FLOAT_TO_16_SCALE) as i16
}

#[inline(always)]
pub fn interleave_71_inner(
    deinterleaved: &DeinterleavedBuffer71,
    interleaved: &mut InterleavedBuffer71,
) {
    let num_samples = deinterleaved.num_samples;
    let dst = &mut interleaved.data[0..num_samples];
    let src_fl = &deinterleaved.data_fl[0..num_samples];
    let src_fr = &deinterleaved.data_fr[0..num_samples];
    let src_fc = &deinterleaved.data_fc[0..num_samples];
    let src_lf = &deinterleaved.data_lf[0..num_samples];
    let src_sl = &deinterleaved.data_sl[0..num_samples];
    let src_sr = &deinterleaved.data_sr[0..num_samples];
    let src_bl = &deinterleaved.data_bl[0..num_samples];
    let src_br = &deinterleaved.data_br[0..num_samples];
    for i in 0..num_samples {
        dst[i].fl = pcm_float_to_16(src_fl[i]);
        dst[i].fr = pcm_float_to_16(src_fr[i]);
        dst[i].fc = pcm_float_to_16(src_fc[i]);
        dst[i].lf = pcm_float_to_16(src_lf[i]);
        dst[i].sl = pcm_float_to_16(src_sl[i]);
        dst[i].sr = pcm_float_to_16(src_sr[i]);
        dst[i].bl = pcm_float_to_16(src_bl[i]);
        dst[i].br = pcm_float_to_16(src_br[i]);
    }
}

// Not used in the article - to ugly
#[inline(always)]
pub fn interleave_71_inner_iter(
    deinterleaved: &DeinterleavedBuffer71,
    interleaved: &mut InterleavedBuffer71,
) {
    for (dst, (fl, (fr, (fc, (lf, (sl, (sr, (bl, br)))))))) in interleaved.data.iter_mut().zip(
        deinterleaved.data_fl.iter().zip(
            deinterleaved.data_fr.iter().zip(
                deinterleaved.data_fc.iter().zip(
                    deinterleaved.data_lf.iter().zip(
                        deinterleaved.data_sl.iter().zip(
                            deinterleaved.data_sr.iter().zip(
                                deinterleaved
                                    .data_bl
                                    .iter()
                                    .zip(deinterleaved.data_br.iter()),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ) {
        dst.fl = pcm_float_to_16(*fl);
        dst.fr = pcm_float_to_16(*fr);
        dst.fc = pcm_float_to_16(*fc);
        dst.lf = pcm_float_to_16(*lf);
        dst.sl = pcm_float_to_16(*sl);
        dst.sr = pcm_float_to_16(*sr);
        dst.bl = pcm_float_to_16(*bl);
        dst.br = pcm_float_to_16(*br);
    }
}

// Not used in the article - godbolt doesn't support izip
#[inline(always)]
pub fn interleave_71_inner_iter_tools(
    deinterleaved: &DeinterleavedBuffer71,
    interleaved: &mut InterleavedBuffer71,
) {
    use itertools::izip;

    for (dst, fl, fr, fc, lf, sl, sr, bl, br) in izip!(
        &mut interleaved.data,
        &deinterleaved.data_fl,
        &deinterleaved.data_fr,
        &deinterleaved.data_fc,
        &deinterleaved.data_lf,
        &deinterleaved.data_sl,
        &deinterleaved.data_sr,
        &deinterleaved.data_bl,
        &deinterleaved.data_br,
    ) {
        dst.fl = pcm_float_to_16(*fl);
        dst.fr = pcm_float_to_16(*fr);
        dst.fc = pcm_float_to_16(*fc);
        dst.lf = pcm_float_to_16(*lf);
        dst.sl = pcm_float_to_16(*sl);
        dst.sr = pcm_float_to_16(*sr);
        dst.bl = pcm_float_to_16(*bl);
        dst.br = pcm_float_to_16(*br);
    }
}

#[target_feature(enable = "avx")]
pub unsafe fn interleave_71_avx(
    deinterleaved: &DeinterleavedBuffer71,
    interleaved: &mut InterleavedBuffer71,
) {
    interleave_71_inner(deinterleaved, interleaved)
}

#[target_feature(enable = "avx2")]
pub unsafe fn interleave_71_avx2(
    deinterleaved: &DeinterleavedBuffer71,
    interleaved: &mut InterleavedBuffer71,
) {
    interleave_71_inner(deinterleaved, interleaved)
}

pub fn interleave_71(deinterleaved: &DeinterleavedBuffer71, interleaved: &mut InterleavedBuffer71) {
    if is_x86_feature_detected!("avx2") {
        unsafe { interleave_71_avx2(deinterleaved, interleaved) }
    } else if is_x86_feature_detected!("avx") {
        unsafe { interleave_71_avx(deinterleaved, interleaved) }
    } else {
        interleave_71_inner(deinterleaved, interleaved)
    }
}

use std::arch::x86_64::*;

#[inline(always)]
pub unsafe fn transpose_and_convert(a: __m256, b: __m256, c: __m256, d: __m256, dst: *mut __m256i) {
    // keep shuffling individual elements inside the vector around to finish interleaving
    let unpack_lo_pd_0 =
        _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(a), _mm256_castps_pd(b)));
    let unpack_lo_pd_1 =
        _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(c), _mm256_castps_pd(d)));

    let unpack_hi_pd_0 =
        _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(a), _mm256_castps_pd(b)));
    let unpack_hi_pd_1 =
        _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(c), _mm256_castps_pd(d)));

    let permute_0 = _mm256_permute2f128_ps(unpack_lo_pd_0, unpack_hi_pd_0, 0b_0010_0000);
    let permute_1 = _mm256_permute2f128_ps(unpack_lo_pd_0, unpack_hi_pd_0, 0b_0011_0001);
    let permute_2 = _mm256_permute2f128_ps(unpack_lo_pd_1, unpack_hi_pd_1, 0b_0010_0000);
    let permute_3 = _mm256_permute2f128_ps(unpack_lo_pd_1, unpack_hi_pd_1, 0b_0011_0001);

    // convert all our values from f32 [-1.0, 1.0] to i32 [-32767, 32767]
    let scale = _mm256_set1_ps(POS_FLOAT_TO_16_SCALE);
    let i32_0 = _mm256_cvtps_epi32(_mm256_mul_ps(permute_0, scale));
    let i32_1 = _mm256_cvtps_epi32(_mm256_mul_ps(permute_1, scale));
    let i32_2 = _mm256_cvtps_epi32(_mm256_mul_ps(permute_2, scale));
    let i32_3 = _mm256_cvtps_epi32(_mm256_mul_ps(permute_3, scale));

    // convert from i32 to i16
    let i16_0 = _mm256_packs_epi32(i32_0, i32_2);
    let i16_1 = _mm256_packs_epi32(i32_1, i32_3);

    // store to destination memory
    _mm256_storeu_si256(dst.offset(0), i16_0);
    _mm256_storeu_si256(dst.offset(2), i16_1);
}

#[target_feature(enable = "avx2")]
pub unsafe fn interleave_71_manual_avx2(
    deinterleaved: &DeinterleavedBuffer71,
    interleaved: &mut InterleavedBuffer71,
) {
    let num_samples = deinterleaved.num_samples;
    assert_eq!(interleaved.data.len(), num_samples);
    assert!(num_samples % 8 == 0);
    let mut dst_base = interleaved.data.as_mut_ptr() as *mut __m256i;
    let mut src_fl_base = deinterleaved.data_fl.as_ptr();
    let mut src_fr_base = deinterleaved.data_fr.as_ptr();
    let mut src_fc_base = deinterleaved.data_fc.as_ptr();
    let mut src_lf_base = deinterleaved.data_lf.as_ptr();
    let mut src_sl_base = deinterleaved.data_sl.as_ptr();
    let mut src_sr_base = deinterleaved.data_sr.as_ptr();
    let mut src_bl_base = deinterleaved.data_bl.as_ptr();
    let mut src_br_base = deinterleaved.data_br.as_ptr();
    for _ in 0..num_samples / 8 {
        let src_fl = _mm256_loadu_ps(src_fl_base);
        let src_fr = _mm256_loadu_ps(src_fr_base);
        let src_fc = _mm256_loadu_ps(src_fc_base);
        let src_lf = _mm256_loadu_ps(src_lf_base);
        let src_sl = _mm256_loadu_ps(src_sl_base);
        let src_sr = _mm256_loadu_ps(src_sr_base);
        let src_bl = _mm256_loadu_ps(src_bl_base);
        let src_br = _mm256_loadu_ps(src_br_base);

        transpose_and_convert(
            _mm256_unpacklo_ps(src_fl, src_fr),
            _mm256_unpacklo_ps(src_fc, src_lf),
            _mm256_unpacklo_ps(src_sl, src_sr),
            _mm256_unpacklo_ps(src_bl, src_br),
            dst_base,
        );
        transpose_and_convert(
            _mm256_unpackhi_ps(src_fl, src_fr),
            _mm256_unpackhi_ps(src_fc, src_lf),
            _mm256_unpackhi_ps(src_sl, src_sr),
            _mm256_unpackhi_ps(src_bl, src_br),
            dst_base.offset(1),
        );

        src_fl_base = src_fl_base.offset(8);
        src_fr_base = src_fr_base.offset(8);
        src_fc_base = src_fc_base.offset(8);
        src_lf_base = src_lf_base.offset(8);
        src_sl_base = src_sl_base.offset(8);
        src_sr_base = src_sr_base.offset(8);
        src_bl_base = src_bl_base.offset(8);
        src_br_base = src_br_base.offset(8);
        dst_base = dst_base.offset(4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_data(start: usize, count: usize) -> Vec<f32> {
        (0..count)
            .map(|i| (i * 8 + start) as f32 / POS_FLOAT_TO_16_SCALE)
            .collect()
    }

    #[test]
    fn reference() {
        let samples = 2;
        let mut interleaved = InterleavedBuffer71::empty(samples);
        let deinterleaved = DeinterleavedBuffer71 {
            num_samples: samples,
            data_fl: test_data(0, samples),
            data_fr: test_data(1, samples),
            data_fc: test_data(2, samples),
            data_lf: test_data(3, samples),
            data_sl: test_data(4, samples),
            data_sr: test_data(5, samples),
            data_bl: test_data(6, samples),
            data_br: test_data(7, samples),
        };
        interleave_71_inner(&deinterleaved, &mut interleaved);
        let expected = InterleavedBuffer71 {
            data: vec![
                InterleavedSample71 {
                    fl: 0,
                    fr: 1,
                    fc: 2,
                    lf: 3,
                    sl: 4,
                    sr: 5,
                    bl: 6,
                    br: 7,
                },
                InterleavedSample71 {
                    fl: 8,
                    fr: 9,
                    fc: 10,
                    lf: 11,
                    sl: 12,
                    sr: 13,
                    bl: 14,
                    br: 15,
                },
            ],
        };

        for (expected, actual) in expected.data.iter().zip(interleaved.data.iter()) {
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn manual_avx2_vs_reference() {
        let samples = 8 * 8;
        let mut interleaved_avx2 = InterleavedBuffer71::empty(samples);
        let mut interleaved = InterleavedBuffer71::empty(samples);
        let deinterleaved = DeinterleavedBuffer71 {
            num_samples: samples,
            data_fl: test_data(0, samples),
            data_fr: test_data(1, samples),
            data_fc: test_data(2, samples),
            data_lf: test_data(3, samples),
            data_sl: test_data(4, samples),
            data_sr: test_data(5, samples),
            data_bl: test_data(6, samples),
            data_br: test_data(7, samples),
        };
        unsafe {
            interleave_71_manual_avx2(&deinterleaved, &mut interleaved_avx2);
        }

        interleave_71_inner(&deinterleaved, &mut interleaved);

        for (expected, actual) in interleaved.data.iter().zip(interleaved_avx2.data.iter()) {
            assert_eq!(expected, actual);
        }
    }
}
