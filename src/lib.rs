#![allow(dead_code)]

use std::default::Default;

#[derive(Default, Copy, Clone)]
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
static NEG_FLOAT_TO_16_SCALE: f32 = 0x8000 as f32;

#[inline(always)]
fn pcm_float_to_16(x: f32) -> i16 {
    // (if x > 0.0 {
    //     x * POS_FLOAT_TO_16_SCALE
    // } else {
    //     x * NEG_FLOAT_TO_16_SCALE
    // }) as i16
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
