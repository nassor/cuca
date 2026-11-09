use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};

use super::write::{RawTensor, write_raw};
use super::*;

fn temp_dir(tag: &str) -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "cuca-gguf-{tag}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn bytes(n: usize, seed: u8) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed))
        .collect()
}

fn s(v: &str) -> Value {
    Value::String(v.into())
}

#[test]
fn a_header_with_types_candle_rejects_opens_and_names_the_tensor_on_use() {
    let dir = temp_dir("types");
    let path = dir.join("m.gguf");
    let mx = bytes(2 * 160 * 17, 1);
    let nl = bytes(4 * 18, 2);
    let i8s = bytes(10, 3);
    let unk = bytes(12, 4);
    let q8: Vec<u8> = QTensor::quantize(
        &Tensor::from_vec(vec![0.5f32; 64], (2, 32), &Device::Cpu).unwrap(),
        GgmlDType::Q8_0,
    )
    .unwrap()
    .data()
    .unwrap()
    .into_owned();
    let meta = [("general.architecture", &s("x"))];
    let tensors = [
        RawTensor {
            name: "mx",
            dims: &[2, 5120],
            dtype: GgufType::Mxfp4,
            data: &mx,
        },
        RawTensor {
            name: "nl",
            dims: &[4, 32],
            dtype: GgufType::Iq4Nl,
            data: &nl,
        },
        RawTensor {
            name: "i8",
            dims: &[2, 5],
            dtype: GgufType::I8,
            data: &i8s,
        },
        RawTensor {
            name: "unk",
            dims: &[3, 4],
            dtype: GgufType::Unknown(16),
            data: &unk,
        },
        RawTensor {
            name: "q8",
            dims: &[2, 32],
            dtype: GgufType::Ggml(GgmlDType::Q8_0),
            data: &q8,
        },
    ];
    write_raw(&mut File::create(&path).unwrap(), &meta, &tensors).unwrap();

    let content = Content::open(&path).unwrap();
    assert_eq!(content.tensor_infos.len(), 5);
    assert_eq!(content.info("mx").unwrap().dtype, GgufType::Mxfp4);
    assert_eq!(content.info("mx").unwrap().shape.dims(), [2, 5120]);
    assert_eq!(content.info("unk").unwrap().dtype, GgufType::Unknown(16));

    let mut r = content.reader();
    assert_eq!(r.bytes("mx").unwrap(), mx);
    assert_eq!(r.bytes("nl").unwrap(), nl);
    assert_eq!(r.bytes("i8").unwrap(), i8s);
    let q = r.tensor("q8", &Device::Cpu).unwrap();
    assert_eq!(q.data().unwrap().as_ref(), q8.as_slice());

    let msg = r.bytes("unk").unwrap_err().to_string();
    assert!(msg.contains("unk") && msg.contains("ggml type 16"), "{msg}");
    let msg = r.tensor("mx", &Device::Cpu).unwrap_err().to_string();
    assert!(msg.contains("`mx`") && msg.contains("MXFP4"), "{msg}");
    let msg = content.info("absent").unwrap_err().to_string();
    assert!(msg.contains("absent"), "{msg}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_tensor_that_runs_past_the_end_of_the_file_is_an_error() {
    let dir = temp_dir("short");
    let path = dir.join("m.gguf");
    let data = bytes(17, 1);
    let meta: [(&str, &Value); 0] = [];
    let t = [RawTensor {
        name: "mx",
        dims: &[32],
        dtype: GgufType::Mxfp4,
        data: &data,
    }];
    write_raw(&mut File::create(&path).unwrap(), &meta, &t).unwrap();
    let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    let len = f.metadata().unwrap().len();
    f.set_len(len - 20).unwrap();
    let content = Content::open(&path).unwrap();
    let msg = content.reader().bytes("mx").unwrap_err().to_string();
    assert!(msg.contains("remaining"), "{msg}");
    let _ = std::fs::remove_dir_all(dir);
}

/// Write a 3-part set: part 1 holds only metadata, parts 2 and 3 one tensor
/// each, with the `split.*` keys llama.cpp writes.
fn write_set(dir: &std::path::Path) -> (PathBuf, Vec<u8>, Vec<u8>) {
    let (a, b) = (bytes(2 * 3 * 17, 5), bytes(4 * 34, 6));
    let name = |no: u32| dir.join(format!("m-{no:05}-of-00003.gguf"));
    let split = |no: u16| {
        vec![
            ("split.no".to_string(), Value::U16(no)),
            ("split.count".to_string(), Value::U16(3)),
            ("split.tensors.count".to_string(), Value::I32(2)),
        ]
    };
    let write_part = |no: u32, extra: Vec<(String, Value)>, tensors: &[RawTensor]| {
        let kv: Vec<(String, Value)> = split(no as u16 - 1).into_iter().chain(extra).collect();
        let refs: Vec<(&str, &Value)> = kv.iter().map(|(k, v)| (k.as_str(), v)).collect();
        write_raw(&mut File::create(name(no)).unwrap(), &refs, tensors).unwrap();
    };
    write_part(1, vec![("general.architecture".into(), s("demo"))], &[]);
    write_part(
        2,
        vec![],
        &[RawTensor {
            name: "a",
            dims: &[2, 96],
            dtype: GgufType::Mxfp4,
            data: &a,
        }],
    );
    write_part(
        3,
        vec![],
        &[RawTensor {
            name: "b",
            dims: &[4, 32],
            dtype: GgufType::Ggml(GgmlDType::Q8_0),
            data: &b,
        }],
    );
    (name(1), a, b)
}

#[test]
fn a_split_set_opens_as_one_content_and_reads_each_part() {
    let dir = temp_dir("set");
    let (first, a, b) = write_set(&dir);
    for named in [first.clone(), dir.join("m-00003-of-00003.gguf")] {
        let content = Content::open(&named).unwrap();
        assert_eq!(content.files().len(), 3);
        assert_eq!(
            content.metadata["general.architecture"]
                .to_string()
                .unwrap(),
            "demo"
        );
        assert_eq!(content.info("a").unwrap().file, 1);
        assert_eq!(content.info("b").unwrap().file, 2);
        let mut r = content.reader();
        assert_eq!(r.bytes("a").unwrap(), a);
        assert_eq!(r.bytes("b").unwrap(), b);
        let (path, at) = content.location(content.info("b").unwrap());
        assert_eq!(path, dir.join("m-00003-of-00003.gguf"));
        assert_eq!(at % 32, 0);
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_missing_part_is_named_and_extra_dirs_are_searched() {
    let dir = temp_dir("missing");
    let (first, a, _) = write_set(&dir);
    let elsewhere = temp_dir("elsewhere");
    std::fs::rename(
        dir.join("m-00003-of-00003.gguf"),
        elsewhere.join("m-00003-of-00003.gguf"),
    )
    .unwrap();
    let msg = Content::open(&first).unwrap_err().to_string();
    assert!(
        msg.contains("part 3 of 3") && msg.contains("m-00003-of-00003.gguf"),
        "{msg}"
    );
    let content = Content::open_with(&first, std::slice::from_ref(&elsewhere)).unwrap();
    assert_eq!(content.reader().bytes("a").unwrap(), a);
    assert_eq!(
        content.location(content.info("b").unwrap()).0,
        elsewhere.join("m-00003-of-00003.gguf")
    );
    let _ = std::fs::remove_dir_all(dir);
    let _ = std::fs::remove_dir_all(elsewhere);
}

/// The reader gives the metadata and tensor bytes candle's reader gives, on a
/// file candle's writer layout produced.
#[test]
fn it_agrees_with_candles_reader_on_the_types_candle_knows() {
    let dir = temp_dir("parity");
    let path = dir.join("m.gguf");
    let mut seed = 1u32;
    let mut vals = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    };
    let make = |v: Vec<f32>, dims: &[usize], d: GgmlDType| {
        QTensor::quantize(
            &Tensor::from_vec(v, dims.to_vec(), &Device::Cpu).unwrap(),
            d,
        )
        .unwrap()
    };
    let tensors = vec![
        ("w.q8", make(vals(64), &[2, 32], GgmlDType::Q8_0)),
        ("w.q4", make(vals(96), &[3, 32], GgmlDType::Q4_0)),
        ("w.f16", make(vals(12), &[3, 4], GgmlDType::F16)),
        ("w.f32", make(vals(5), &[5], GgmlDType::F32)),
        ("w.q6k", make(vals(512), &[2, 256], GgmlDType::Q6K)),
    ];
    let meta = [
        ("general.architecture", s("t")),
        ("t.n", Value::U32(7)),
        ("t.f", Value::F32(0.25)),
        ("t.b", Value::Bool(true)),
        ("t.arr", Value::Array(vec![Value::I32(1), Value::I32(-2)])),
        ("t.sarr", Value::Array(vec![s("a"), s("bc")])),
        ("t.u64", Value::U64(1 << 40)),
    ];
    let meta_refs: Vec<(&str, &Value)> = meta.iter().map(|(k, v)| (*k, v)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> = tensors.iter().map(|(k, v)| (*k, v)).collect();
    write(&mut File::create(&path).unwrap(), &meta_refs, &tensor_refs).unwrap();

    // The candle reader is the oracle here and nowhere else.
    let theirs = candle_core::quantized::gguf_file::Content::read(&mut BufReader::new(
        File::open(&path).unwrap(),
    ))
    .unwrap();
    let ours = Content::open(&path).unwrap();
    assert_eq!(ours.metadata.len(), theirs.metadata.len());
    for (k, v) in &theirs.metadata {
        assert_eq!(format!("{v:?}"), format!("{:?}", ours.metadata[k]), "{k}");
    }
    assert_eq!(ours.tensor_infos.len(), theirs.tensor_infos.len());
    let mut r = ours.reader();
    for (name, qt) in &tensors {
        let info = &ours.tensor_infos[*name];
        assert_eq!(info.dtype, GgufType::Ggml(qt.dtype()), "{name}");
        assert_eq!(info.shape.dims(), qt.shape().dims(), "{name}");
        assert_eq!(info.offset, theirs.tensor_infos[*name].offset, "{name}");
        assert_eq!(
            ours.location(info).1,
            theirs.tensor_data_offset + theirs.tensor_infos[*name].offset
        );
        assert_eq!(
            r.bytes(name).unwrap(),
            qt.data().unwrap().as_ref(),
            "{name}"
        );
        let back = r.tensor(name, &Device::Cpu).unwrap();
        assert_eq!(back.data().unwrap().as_ref(), qt.data().unwrap().as_ref());
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn split_names_parse_and_render() {
    let n = SplitName::parse("m-00002-of-00006.gguf").unwrap();
    assert_eq!((n.base.as_str(), n.no, n.count), ("m", 2, 6));
    assert_eq!(n.part_name(6), "m-00006-of-00006.gguf");
    assert!(SplitName::parse("m-00007-of-00006.gguf").is_none());
    assert!(SplitName::parse("m.gguf").is_none());
}

#[test]
fn split_names_parse_and_list_their_parts() {
    let s = SplitName::parse("Mistral-Q4_0-00002-of-00003.gguf").unwrap();
    assert_eq!(
        s,
        SplitName {
            base: "Mistral-Q4_0".into(),
            no: 2,
            count: 3
        }
    );
    assert_eq!(s.part_name(1), "Mistral-Q4_0-00001-of-00003.gguf");
    assert_eq!(
        s.part_names(),
        [
            "Mistral-Q4_0-00001-of-00003.gguf",
            "Mistral-Q4_0-00002-of-00003.gguf",
            "Mistral-Q4_0-00003-of-00003.gguf"
        ]
    );
    for bad in [
        "model.gguf",
        "m-1-of-2.gguf",
        "m-00000-of-00002.gguf",
        "m-00003-of-00002.gguf",
        "-00001-of-00002.gguf",
        "m-00001-of-00002.bin",
    ] {
        assert!(SplitName::parse(bad).is_none(), "{bad}");
    }
}
