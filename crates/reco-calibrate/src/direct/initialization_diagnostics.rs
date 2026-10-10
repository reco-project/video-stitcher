//! JSON adapter for frozen placement-initialization comparisons.

use super::*;
use serde_json::{Value, json};
use std::sync::atomic::AtomicBool;

// Keep the established comparison protocol outside the typed implementation.
pub(crate) fn initialize(request: &Value, interrupted: &AtomicBool) -> Result<Value, String> {
    let millis = request
        .get("time_limit_ms")
        .map_or(Some(30_000), Value::as_u64)
        .filter(|v| (1..=30_000).contains(v))
        .ok_or("invalid initialization time limit")?;
    let control = Control {
        interrupted,
        deadline: Instant::now() + Duration::from_millis(millis),
    };
    control.check()?;
    let source: Vec<[f64; 2]> =
        serde_json::from_value(request["source"].clone()).map_err(|e| e.to_string())?;
    let target: Vec<[f64; 2]> =
        serde_json::from_value(request["target"].clone()).map_err(|e| e.to_string())?;
    let dimensions: [[usize; 2]; 2] =
        serde_json::from_value(request["dimensions"].clone()).map_err(|e| e.to_string())?;
    // Check layout before decoding the document, matching the diagnostic protocol.
    if !(15..=2000).contains(&source.len())
        || source.len() != target.len()
        || dimensions
            .iter()
            .flatten()
            .any(|v| !(2..=16_384).contains(v))
    {
        return Err("invalid initialization layout".into());
    }
    let seed = serde_json::from_value(request["seed"].clone()).map_err(|e| e.to_string())?;
    let r = initialize_native(seed, &source, &target, dimensions, &control)?;
    let stages: Vec<_> = r
        .stages
        .into_iter()
        .map(|s| {
            json!({
                "scale":s.scale, "status":s.status, "iterations":s.iterations,
                "cost":s.cost, "calibration":s.calibration,
            })
        })
        .collect();
    Ok(json!({
        "calibration":r.calibration,"x":r.x,"stages":stages,"seconds":r.elapsed.as_secs_f64()
    }))
}
