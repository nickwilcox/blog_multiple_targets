#![allow(dead_code)]

use std::default::Default;

pub type Sample = u16;

#[derive(Default, Copy, Clone)]
pub struct InterleavedSample71 {
    fl: Sample,
    fr: Sample,
    fc: Sample,
    lf: Sample,
    sl: Sample,
    sr: Sample,
    rl: Sample,
    rr: Sample,
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
    data_fl: Vec<Sample>,
    data_fr: Vec<Sample>,
    data_fc: Vec<Sample>,
    data_lf: Vec<Sample>,
    data_sl: Vec<Sample>,
    data_sr: Vec<Sample>,
    data_rl: Vec<Sample>,
    data_rr: Vec<Sample>,
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
        dst[i].fl = src_fl[i];
        dst[i].fr = src_fr[i];
        dst[i].fc = src_fc[i];
        dst[i].lf = src_lf[i];
        dst[i].sl = src_sl[i];
        dst[i].sr = src_sr[i];
        dst[i].rl = src_rl[i];
        dst[i].rr = src_rr[i];
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
