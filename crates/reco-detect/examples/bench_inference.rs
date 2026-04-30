use ort::{session::Session, session::builder::GraphOptimizationLevel, value::TensorRef};
use std::time::Instant;

fn main() {
    let model_path = std::env::args()
        .nth(1)
        .expect("Usage: bench_inference <model.onnx>");

    for threads in [0u16, 2, 4, 8] {
        let mut builder = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap();

        if threads > 0 {
            builder = builder.with_intra_threads(threads as usize).unwrap();
        }

        // Try DirectML on Windows
        #[cfg(target_os = "windows")]
        let mut builder = {
            match builder.with_execution_providers([ort::ep::DirectML::default().build()]) {
                Ok(b) => {
                    if threads == 0 {
                        println!("DirectML enabled");
                    }
                    b
                }
                Err(e) => {
                    if threads == 0 {
                        println!("DirectML unavailable, using CPU");
                    }
                    e.recover()
                }
            }
        };

        let mut session = builder.commit_from_file(&model_path).unwrap();

        let input = &session.inputs()[0];
        let shape = match input.dtype() {
            ort::value::ValueType::Tensor { shape, .. } => shape.clone(),
            _ => panic!("unexpected input type"),
        };
        let h = shape[2] as usize;
        let w = shape[3] as usize;
        let sz = 3 * h * w;
        let data: Vec<f32> = vec![0.5; sz];

        let label = if threads == 0 {
            "default".to_string()
        } else {
            format!("{threads} threads")
        };

        // Warmup
        for _ in 0..3 {
            let t = TensorRef::from_array_view(([1usize, 3, h, w], data.as_slice())).unwrap();
            let _ = session.run(ort::inputs![t]).unwrap();
        }

        // Bench
        let n = 10;
        let mut times = Vec::with_capacity(n);
        for _ in 0..n {
            let t = TensorRef::from_array_view(([1usize, 3, h, w], data.as_slice())).unwrap();
            let t0 = Instant::now();
            let _ = session.run(ort::inputs![t]).unwrap();
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }

        let avg: f64 = times.iter().sum::<f64>() / n as f64;
        let min = times.iter().cloned().fold(f64::MAX, f64::min);
        println!("{label:15}: avg={avg:7.1}ms  min={min:6.1}ms  (input {h}x{w})");
    }
}
