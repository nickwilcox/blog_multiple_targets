#![allow(dead_code)]

use std::default::Default;

// In the following code we use the following acronyms for conciseness
// FL - front left
// FR - front right
// FC - front center, also just called center
// LF - low frequency, also called low frequency effects, or sub-woofer
// SL - surround left
// SR - surround right
// RL - rear left
// RR - read right

#[derive(Default, Copy, Clone, Eq, PartialEq, Debug)]
pub struct InterleavedSample71 {
    fl: i16,
    fr: i16,
    fc: i16,
    lf: i16,
    sl: i16,
    sr: i16,
    rl: i16,
    rr: i16,
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
    data_rl: Vec<f32>,
    data_rr: Vec<f32>,
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
            data_rl: vec![Default::default(); num_samples],
            data_rr: vec![Default::default(); num_samples],
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
    let src_rl = &deinterleaved.data_rl[0..num_samples];
    let src_rr = &deinterleaved.data_rr[0..num_samples];
    for i in 0..num_samples {
        dst[i].fl = pcm_float_to_16(src_fl[i]);
        dst[i].fr = pcm_float_to_16(src_fr[i]);
        dst[i].fc = pcm_float_to_16(src_fc[i]);
        dst[i].lf = pcm_float_to_16(src_lf[i]);
        dst[i].sl = pcm_float_to_16(src_sl[i]);
        dst[i].sr = pcm_float_to_16(src_sr[i]);
        dst[i].rl = pcm_float_to_16(src_rl[i]);
        dst[i].rr = pcm_float_to_16(src_rr[i]);
    }
}

#[inline(always)]
pub fn interleave_71_inner_iter(
    deinterleaved: &DeinterleavedBuffer71,
    interleaved: &mut InterleavedBuffer71,
) {
    for (dst, (fl, (fr, (fc, (lf, (sl, (sr, (rl, rr)))))))) in interleaved.data.iter_mut().zip(
        deinterleaved.data_fl.iter().zip(
            deinterleaved.data_fr.iter().zip(
                deinterleaved.data_fc.iter().zip(
                    deinterleaved.data_lf.iter().zip(
                        deinterleaved.data_sl.iter().zip(
                            deinterleaved.data_sr.iter().zip(
                                deinterleaved
                                    .data_rl
                                    .iter()
                                    .zip(deinterleaved.data_rr.iter()),
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
        dst.rl = pcm_float_to_16(*rl);
        dst.rr = pcm_float_to_16(*rr);
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
unsafe fn _mm256_unpacklo_pd_ps(a: __m256, b: __m256) -> __m256 {
    *(&_mm256_unpacklo_pd(
        *(&a as *const _ as *const __m256d),
        *(&b as *const _ as *const __m256d),
    ) as *const _ as *const __m256)
}

#[inline(always)]
unsafe fn _mm256_unpackhi_pd_ps(a: __m256, b: __m256) -> __m256 {
    *(&_mm256_unpackhi_pd(
        *(&a as *const _ as *const __m256d),
        *(&b as *const _ as *const __m256d),
    ) as *const _ as *const __m256)
}

#[target_feature(enable = "avx2")]
pub unsafe fn interleave_71_manual_avx2(
    deinterleaved: &DeinterleavedBuffer71,
    interleaved: &mut InterleavedBuffer71,
) {
    let num_samples = deinterleaved.num_samples;
    let _dst_debug = interleaved.data.as_ptr() as *const i16;
    let mut dst_base = interleaved.data.as_mut_ptr() as *mut __m256i;
    let mut src_fl_base = deinterleaved.data_fl.as_ptr();
    let mut src_fr_base = deinterleaved.data_fr.as_ptr();
    let mut src_fc_base = deinterleaved.data_fc.as_ptr();
    let mut src_lf_base = deinterleaved.data_lf.as_ptr();
    let mut src_sl_base = deinterleaved.data_sl.as_ptr();
    let mut src_sr_base = deinterleaved.data_sr.as_ptr();
    let mut src_rl_base = deinterleaved.data_rl.as_ptr();
    let mut src_rr_base = deinterleaved.data_rr.as_ptr();
    let scale = _mm256_set1_ps(POS_FLOAT_TO_16_SCALE);
    for _ in 0..num_samples / 8 {
        let src_fl = _mm256_loadu_ps(src_fl_base);
        let src_fr = _mm256_loadu_ps(src_fr_base);
        let src_fc = _mm256_loadu_ps(src_fc_base);
        let src_lf = _mm256_loadu_ps(src_lf_base);
        let src_sl = _mm256_loadu_ps(src_sl_base);
        let src_sr = _mm256_loadu_ps(src_sr_base);
        let src_rl = _mm256_loadu_ps(src_rl_base);
        let src_rr = _mm256_loadu_ps(src_rr_base);

        {
            let unpack_lo_ps_0 = _mm256_unpacklo_ps(src_fl, src_fr);
            let unpack_lo_ps_1 = _mm256_unpacklo_ps(src_fc, src_lf);
            let unpack_lo_ps_2 = _mm256_unpacklo_ps(src_sl, src_sr);
            let unpack_lo_ps_3 = _mm256_unpacklo_ps(src_rl, src_rr);

            {
                let unpack_lo_pd_0 = _mm256_unpacklo_pd_ps(unpack_lo_ps_0, unpack_lo_ps_1);
                let unpack_lo_pd_1 = _mm256_unpacklo_pd_ps(unpack_lo_ps_2, unpack_lo_ps_3);

                let unpack_hi_pd_0 = _mm256_unpackhi_pd_ps(unpack_lo_ps_0, unpack_lo_ps_1);
                let unpack_hi_pd_1 = _mm256_unpackhi_pd_ps(unpack_lo_ps_2, unpack_lo_ps_3);

                let permute_0 =
                    _mm256_permute2f128_ps(unpack_lo_pd_0, unpack_hi_pd_0, 0b_0010_0000);
                let permute_1 =
                    _mm256_permute2f128_ps(unpack_lo_pd_0, unpack_hi_pd_0, 0b_0011_0001);
                let permute_2 =
                    _mm256_permute2f128_ps(unpack_lo_pd_1, unpack_hi_pd_1, 0b_0010_0000);
                let permute_3 =
                    _mm256_permute2f128_ps(unpack_lo_pd_1, unpack_hi_pd_1, 0b_0011_0001);

                let i32_0 = _mm256_cvtps_epi32(_mm256_mul_ps(permute_0, scale));
                let i32_1 = _mm256_cvtps_epi32(_mm256_mul_ps(permute_1, scale));
                let i32_2 = _mm256_cvtps_epi32(_mm256_mul_ps(permute_2, scale));
                let i32_3 = _mm256_cvtps_epi32(_mm256_mul_ps(permute_3, scale));

                let i16_0 = _mm256_packs_epi32(i32_0, i32_2);
                let i16_1 = _mm256_packs_epi32(i32_1, i32_3);

                _mm256_storeu_si256(dst_base.offset(0), i16_0);
                _mm256_storeu_si256(dst_base.offset(2), i16_1);
            }
        }
        {
            let unpack_hi_ps_0 = _mm256_unpackhi_ps(src_fl, src_fr);
            let unpack_hi_ps_1 = _mm256_unpackhi_ps(src_fc, src_lf);
            let unpack_hi_ps_2 = _mm256_unpackhi_ps(src_sl, src_sr);
            let unpack_hi_ps_3 = _mm256_unpackhi_ps(src_rl, src_rr);

            {
                let unpack_lo_pd_0 = _mm256_unpacklo_pd_ps(unpack_hi_ps_0, unpack_hi_ps_1);
                let unpack_lo_pd_1 = _mm256_unpacklo_pd_ps(unpack_hi_ps_2, unpack_hi_ps_3);

                let unpack_hi_pd_0 = _mm256_unpackhi_pd_ps(unpack_hi_ps_0, unpack_hi_ps_1);
                let unpack_hi_pd_1 = _mm256_unpackhi_pd_ps(unpack_hi_ps_2, unpack_hi_ps_3);

                let permute_0 =
                    _mm256_permute2f128_ps(unpack_lo_pd_0, unpack_hi_pd_0, 0b_0010_0000);
                let permute_1 =
                    _mm256_permute2f128_ps(unpack_lo_pd_0, unpack_hi_pd_0, 0b_0011_0001);
                let permute_2 =
                    _mm256_permute2f128_ps(unpack_lo_pd_1, unpack_hi_pd_1, 0b_0010_0000);
                let permute_3 =
                    _mm256_permute2f128_ps(unpack_lo_pd_1, unpack_hi_pd_1, 0b_0011_0001);

                let i32_0 = _mm256_cvtps_epi32(_mm256_mul_ps(permute_0, scale));
                let i32_1 = _mm256_cvtps_epi32(_mm256_mul_ps(permute_1, scale));
                let i32_2 = _mm256_cvtps_epi32(_mm256_mul_ps(permute_2, scale));
                let i32_3 = _mm256_cvtps_epi32(_mm256_mul_ps(permute_3, scale));

                let i16_0 = _mm256_packs_epi32(i32_0, i32_2);
                let i16_1 = _mm256_packs_epi32(i32_1, i32_3);

                _mm256_storeu_si256(dst_base.offset(1), i16_0);
                _mm256_storeu_si256(dst_base.offset(3), i16_1);
            }
        }

        src_fl_base = src_fl_base.offset(8);
        src_fr_base = src_fr_base.offset(8);
        src_fc_base = src_fc_base.offset(8);
        src_lf_base = src_lf_base.offset(8);
        src_sl_base = src_sl_base.offset(8);
        src_sr_base = src_sr_base.offset(8);
        src_rl_base = src_rl_base.offset(8);
        src_rr_base = src_rr_base.offset(8);
        dst_base = dst_base.offset(4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_data(start: usize, count: usize) -> Vec<f32> {
        (0..count)
            .map(|i| (i * count + start) as f32 / POS_FLOAT_TO_16_SCALE)
            .collect()
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
            data_rl: test_data(6, samples),
            data_rr: test_data(7, samples),
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
