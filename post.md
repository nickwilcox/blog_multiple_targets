
## Introduction

In the [previous article]({% post_url 2020-05-16-autovec %}) on auto-vectorization we treated instructions are SIMD (*S*ingle *I*instruction *M*ultiple *D*ata) or not-SIMD. We also assumed that the *M*utiple meant four values at a time. That was true for way we wrote and compiled our code in that article, but we're going to expand beyond that. We're going to see there is a progression of *instruction families*, with new releases of CPU's from Intel and AMD supporting new instructions we can use to increase the performance of our code.

In this article we're going to:
* Look at the compiler output when targeting the different SIMD instruction set families.
* Benchmark the different instruction sets.
* Look at how we can structure our Rust code to support compiling and then selecting at runtime the instruction set to use.

## A Background on the Different SIMD Instruction Sets on x86 CPUs

Technically the history of SIMD on x86 CPU's starts with the MMX family of instructions on the Pentium in 1997. But we can actually skip the early history and go straight to the SSE2 family. The reason this family is so important is that it's the only one guaranteed to be supported by a 64-bit X86 CPU. It supports a wide variety of mathematical and logical operations of 4 floating point values and as well as packed 8, 16, 32, and 64bit integer types.

All the SIMD from the previous article was limited to SSE2 instructions as the default settings for the compiler is to produce code that is compatible with all 64-bit X86 CPU's.

There have been many instruction families released after SSE2. The ones that followed directly are SSE3, SSSE3, SSE4.1, and SSE4.1. As the naming suggests these are all incremental upgrades. They add new instructions that make it easier to optimise a broader variety of applications using SIMD. But the number of floating point values processed by each instruction remained at four.

When Intel introduced AVX in 2011, it was a major change as it introduced registers that are 256 bits wide. Meaning that could support up to eight floats. AVX2 was introduced in 2016, and while it didn't increase the size of the registers, it did add new instructions that made writing SIMD code easier.

The newest instructions added are in the AVX512 family. As the name suggests it increases the register size of 512 bits, doubling the number of floats that can be processed in a single instruction to sixteen.

### Hardware surveys

In evaluating what instructions families we want to support, it can be useful to look at data sets for a large number of users.

The [Steam Hardware Survey](https://store.steampowered.com/hwsurvey/Steam-Hardware-Software-Survey-Welcome-to-Steam) is the best such survey.

| Instruction Family | Percentage of Steam Users with Computers that Support It (April 2020) |
|--------------------|-----------------------------------------------------------------------|
| SSE2               | 100%                                                                  |
| SSE3               | 100.00%                                                               |
| SSSE3              | 98.52%                                                                |
| SSE4.1             | 97.80%                                                                |
| SSE4.2             | 97.05%                                                                |
| AVX                | 92.25%                                                                |
| AVX2               | 73.88%                                                                |
| AVX512F            | 0.30%                                                                 |

We can see that between SSE2 first appearing on CPU's in 2000 and being mandatory on 64bit it's support level in 2020 is at 100%. But the trend is the newer the instruction family the lower it's support in CPU's in the wild.

Be aware that this survey largely reflects western gamers and might not capture other types of users well.

## The New Test Function

To test the compiler we're going to use a task a little tougher than the one used in the previous article.

When we store digital audio data with multiple channels, such as stereo with two channels or full surround with eight channels, it can be done two ways. Either interleaved or de-interleaved. De-interleaved means that each channel has it's own slice of memory. Interleaved means the samples are adjacent in memory and there is only a single slice of memory.

Interleaved audio is used in audio file formats such as WAV, or when sending audio to output devices. De-interleaved audio is often used when processing audio in memory.

We can get a better understanding if we look at how we would code 7.1 surround sound in the two representations with Rust types:
```rust
// In all the following code we use these acronyms for our channel names for conciseness
// FL - front left
// FR - front right
// FC - front center, also just called center
// LF - low frequency, also called low frequency effects, or sub-woofer
// SL - surround left
// SR - surround right
// RL - rear left
// RR - rear right

// A single sample
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
```

You might notice that the interleaved audio represents each channel as a signed 16bit number, while the de-interleaved using floating point. This is again because 16bit is mostly used in storing audio file formats such as WAV, and floating point is used in memory when applying effects or mixing.

The function we're going to explore the compilation of takes a de-interleaved buffer and transforms it to an interleaved buffer. The basic structure is to loop over the samples and for each one: load, re-arrange, convert from float to 16bit, and then store.

```rust
static POS_FLOAT_TO_16_SCALE: f32 = 0x7fff as f32;

#[inline(always)]
fn pcm_float_to_16(x: f32) -> i16 {
    (x * POS_FLOAT_TO_16_SCALE) as i16
}

pub fn interleave_71(
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
```

***N.B** I decided for readability to use the index version rather than iterators and multiple nested `zip` functions. `izip!` from the [itertools](https://docs.rs/itertools/0.9.0/itertools/) crate is a more readable versions but is not available in the compiler explorer. All these options produce equivalent compiler output*

## Controlling the Compiler

All the assembler output from the compiler presented in the previous article was produced by the Rust compiler with the default settings for a release build on x86_64. As we said earlier x86 64 bit has guaranteed support for SSE2, so this is the highest SIMD instruction family the compiler used when compiling with these settings.

If we use those same compiler settings and compile this new function, we can see in the [assembly output](https://godbolt.org/z/qFGbbh) that our lessons from the previous article have carried over and that the compiler has produced SIMD code for this loop.

```nasm
.LBB0_40:
        movups          xmm0, xmmword ptr [rdx + 4*rax]
        mulps           xmm0, xmm8
        cvttps2dq       xmm7, xmm0
        movups          xmm0, xmmword ptr [r9 + 4*rax]
        mulps           xmm0, xmm8
        cvttps2dq       xmm9, xmm0
        movups          xmm0, xmmword ptr [r10 + 4*rax]
        mulps           xmm0, xmm8
        cvttps2dq       xmm13, xmm0
        movups          xmm0, xmmword ptr [r11 + 4*rax]
        mulps           xmm0, xmm8
        cvttps2dq       xmm11, xmm0
        movups          xmm0, xmmword ptr [r14 + 4*rax]
        mulps           xmm0, xmm8
        cvttps2dq       xmm15, xmm0
        movups          xmm0, xmmword ptr [r15 + 4*rax]
        mulps           xmm0, xmm8
        cvttps2dq       xmm10, xmm0
        movups          xmm0, xmmword ptr [r12 + 4*rax]
        mulps           xmm0, xmm8
        cvttps2dq       xmm2, xmm0
        movups          xmm0, xmmword ptr [rbx + 4*rax]
        mulps           xmm0, xmm8
        cvttps2dq       xmm12, xmm0
```

There are the signs of SIMD we discussed before. Instructions such as `mulps` which is short for **Mul**tiply **P**acked **S**ingle-precision-float, packed being another term for operating on four values at once.

There are other features to notice in this disassembly. If we look at the registers passed as arguments to each instruction we see they all follow the naming conventions `xmm<NUMBER>`. Registers following this naming convention are called the SSE registers and hold up to four floats.

### Telling the Compiler to use Newer Instruction Families


Using the godbolt compiler explorer we can pass additional compiler arguments. In addition to turning on optimization (the '-O' argument) we can allow the compiler to use additional instruction families.

The following is snippet from the start of the loop of compiler output with the arguments `-O -C target-feature=+avx`

```nasm
.LBB0_40:
        vmovaps         ymm6, ymmword ptr [rip + .LCPI0_0]
        vmulps          ymm0, ymm6, ymmword ptr [rdx + 4*rax]
        vcvttps2dq      ymm0, ymm0
        vextractf128    xmm1, ymm0, 1
        vpackssdw       xmm11, xmm0, xmm1
        vmulps          ymm0, ymm6, ymmword ptr [r9 + 4*rax]
        vcvttps2dq      ymm0, ymm0
        vextractf128    xmm2, ymm0, 1
        vmulps          ymm3, ymm6, ymmword ptr [r10 + 4*rax]
        vpackssdw       xmm12, xmm0, xmm2
        vcvttps2dq      ymm0, ymm3
        vextractf128    xmm3, ymm0, 1
        vpackssdw       xmm13, xmm0, xmm3
        vmulps          ymm0, ymm6, ymmword ptr [r11 + 4*rax]
        vcvttps2dq      ymm0, ymm0
        vextractf128    xmm4, ymm0, 1
        vpackssdw       xmm14, xmm0, xmm4
```

[Full Output](https://godbolt.org/z/Mq534D)

The first thing to notice is that we see that in addition to the SSE registers we see registers with the naming convention `ymm<NUMBER>`. As we discussed earlier in the article AVX added new 256 bit wide registers capable of storing up to eight floats. Now we see how they appear in our assembly. Not surprisingly registers starting with `ymm` are refered to as AVX registers.

When we see a SIMD instruction such as `mulps` we now need to look at the registers that are being passed to it. If we see `xmm` registers we know it's performing four multiplies. If the register arguments are `ymm` then it's performing eight multiplies.

I'm going to skip over the fact that in the AVX compiled output `mulps` is actually `vmulps`. This is a result of VEX encoding and not relevant to these discussions.

If we change our compiler arguments to `-O -C target-feature=+avx2` we can view the [resulting output](https://godbolt.org/z/jLPy2c). It might be hard to spot the difference at first. Not a lot has changed. AVX2 was an iteration of AVX. It added no new register types, only new instructions. If you scan the instructions you can see that the compiler has generated instructions such as `vpblendvb` and `vinserti128`. Consulting the (Intel Intrinsics Guide)[https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=AVX2] show these are part of the new instructions for AVX2.

## Fine Grained Control

Compiling our entire program so it only supports a subset of available CPU's may not be feasible. But there are alternatives to compiler flags we can use. 

If it's possible to pay the cost of a run-time check we can use the built in `target_feature` attribute and `is_x86_feature_detected` macro.

If we re-write our code
```rust
#[inline(always)]
fn interleave_71_inner(
    deinterleaved: &DeinterleavedBuffer71,
    interleaved: &mut InterleavedBuffer71,
) {
    // the body of this function is the same as the earlier interleave_71
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
```

To break what we've done to the code:
Firstly we've renamed our function `interleave_71` to `interleave_71_inner` and forced it to be inlined everywhere it's used.

Secondly we added two new functions `interleave_71_avx` and `interleave_71_avx2`. They use the `target_feature` attribute to tell the compiler that for this function it's allowed to assume the CPU is capable of the given instruction family. Because the function inlines the contents of `interleave_71_inner` this results in us getting a version of our code with the allowed instructions. Because calling this code on a CPU that doesn't support the instructions in undefined behavior we've had to mark our functions as unsafe.

Finally we have a new public function `interleave_71`.It uses the `is_x86_feature_detected` macro to check the CPU, which let's it know if it's safe to call one of our higher instruction family compiled functions. If it doesn't detect the instructions are available it will fall back to the default compiled version. This allows us to present a safe function that doesn't allow undefined behavior. We've arrange the function in descending order of date of introduction of the instruction family.

Be aware that this pattern should only be applied to functions that perform enough work to justify the runtime check. Apply it on the level of only a few short instructions may be a net loss in performance.

I never used them but there are proc macro crates to automate this pattern available: [cfg-specialize](https://github.com/alexcrichton/cfg-specialize) and [runtime-target-feature-rs](https://github.com/parched/runtime-target-feature-rs).

## Does it Matter (is it Faster?)

We've learned about the different instructions families. We've seen that we can either compile our whole application to only target a subset of available CPU's or add extra code to detect support at runtime.

Do the performance gains justify the cost? It's possible to set up benchmark of all the variations on the same CPU for comparison. I've also included a benchmark using `-O -C no-vectorize-loops` to get an idea of the overall benefits of SIMD.

|                          | Time for 100,000 Samples |
|--------------------------|--------------------------|
| vectorize-loops disabled | 321 μs                   |
| SSE2                     | 215 μs                   |
| AVX                      | 245 μs                   |
| AV2                      | 207 μs                   |

Unfortunately the answer seems to be no. With the current version of Rust (1.43.0) it seems that the AVX performance is worse than simple SSE2. AVX2 is a slight win, but in testing with machines other than my main laptop it was also slower than the SSE laptop.

## Comparing the Compiler's AVX2 Code to Hand Written

To judge the assembly output by the compiler we're going to again write an version of the function written by hand using the intrinsics in the Rust's `core::arch::x86_64` module.

```rust
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
```

The things to note about this code is that all the intrinsics we use start with `_mm256` to indicate they are from the AVX or AVX2 instruction family. We've also had to keep `#[target_feature(enable = "avx2")]` attribute on our function to ensure the compiler generates the correct code for our intrinsics.

|                          | Time for 100,000 Samples |
|--------------------------|--------------------------|
| vectorize-loops disabled | 321 μs                   |
| SSE2                     | 215 μs                   |
| AVX                      | 245 μs                   |
| AV2                      | 207 μs                   |
| **Hand Written AVX2**    | **104 μs**               |

As we might expect from an instruction set able to process eight values at a time instead of four, the hand written AVX2 code is twice as fast as the SSE2 code.

We're not going to fully break down what the compiler has produced when auto-vectorizing for AVX2. But we can get do a brief comparison against the hand written version.

If we look at hand written version of AVX2 it's verbose but straight forward in places. 

Remember the basic structure is to loop over the samples and for each one: load, re-arrange, convert from float to 16bit, and then store. The intrinsics version does eight samples at a time.

`_mm256_loadu_ps` and `_mm256_storeu_si256` are load and store. `_mm256_mul_ps`, `_mm256_cvtps_epi32`, and `_mm256_packs_epi32` are used to convert from floating point values in the range (-1.0, 1.0) to signed 16 bit integers.

The hard part of the code to follow is use of `_mm256_permute2f128_ps` and the various unpack instructions such as `_mm256_unpacklo_ps` to re-arrange the data, the actual interleaving part. If we compare this part of the hand written to the compiler generated assembly we can see where the compiler has produced sub-optimal code.

## Conclusion

AVX and AVX2 are supported by large percentage of CPU's right now. We can target them at compile time or runtime and get a boost in the performance of our code. However the compilers may be lacking in there ability to auto-vectorize code for these instruction families. Benchmarking and examination of the compiler output are still required in performance critical situations.

## Sources

All the source code for this article can be found [on github](https://github.com/nickwilcox/blog_multiple_target)

