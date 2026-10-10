//! JSON adapter for frozen numerical comparisons; not a calibration API.

use super::*;
use serde_json::{Value, json};

fn number(value: &Value) -> Result<f64, String> {
    value
        .as_f64()
        .filter(|x| x.is_finite())
        .ok_or("expected finite number".into())
}

// Diagnostic adapter; not part of the native calibration call path.
pub fn solve(
    request: &Value,
    stage: &Stage,
    pool: &rayon::ThreadPool,
    interrupted: &AtomicBool,
) -> Result<Value, String> {
    let method = match request["method"].as_str().ok_or("missing method")? {
        "lm" => Method::LeastSquares,
        "trf" => Method::LeastSquares,
        "slsqp" => Method::Constrained,
        _ => return Err("unknown native solver".into()),
    };
    let duration = match request.get("time_limit_ms") {
        None => Duration::from_secs(30),
        Some(value) => Duration::from_millis(
            value
                .as_u64()
                .filter(|ms| (1..=30_000).contains(ms))
                .ok_or("invalid time limit")?,
        ),
    };
    let control = Control {
        interrupted,
        deadline: Instant::now() + duration,
    };
    control.check()?;
    let seed = serde_json::from_value(request["seed"].clone()).map_err(|e| e.to_string())?;
    let x = serde_json::from_value(request["x"].clone()).map_err(|e| e.to_string())?;
    let options = SolveOptions {
        method,
        budget: request["budget"].as_u64().ok_or("invalid solver budget")?,
        tolerance: request["tolerance"].as_f64().unwrap_or(1e-5),
        difference_step: difference_step(request)?,
        active_penalty: request["active_penalty"].as_bool().unwrap_or(false),
        optics: [true; 2],
    };
    let r = solve_native(seed, x, options, stage, pool, &control)?;
    let mut output = encode_solution(&r);
    output["active_penalty"] = json!(options.active_penalty);
    output["difference_step"] = json!(options.difference_step);
    Ok(output)
}

pub(crate) fn encode_solution(r: &Solution) -> Value {
    json!({
        "x":r.x,"cost":r.cost,"status":r.status,"success":r.success,"iterations":r.iterations,
        "seconds":r.elapsed.as_secs_f64(),"calibration":r.calibration,"margin":r.margin,
        "sensor_boundary_active":r.sensor_boundary_active,
        "residuals":r.residuals,"jacobians":r.jacobians,
        "derivative_seconds":r.derivative_elapsed.as_secs_f64()
    })
}

pub fn inspect(
    request: &Value,
    stage: &Stage,
    pool: &rayon::ThreadPool,
) -> Result<(Value, Vec<f64>), String> {
    let interrupted = AtomicBool::new(false);
    let control = Control {
        interrupted: &interrupted,
        deadline: Instant::now() + Duration::from_secs(30),
    };
    let x: Vec<f64> = serde_json::from_value(request["x"].clone()).map_err(|e| e.to_string())?;
    let mut model = Model::new(
        serde_json::from_value(request["seed"].clone()).map_err(|e| e.to_string())?,
        stage,
        pool,
        request["sensor"].as_bool().unwrap_or(true),
        x.len(),
        &control,
    )?;
    model.active_penalty_derivative = request["active_penalty"].as_bool().unwrap_or(false);
    model.difference_step = difference_step(request)?;
    let values = model.values(&x)?;
    let rows = values.len();
    let (values, cols) = if request["jacobian"].as_bool() == Some(true) {
        (model.columns(&x)?.into_iter().flatten().collect(), x.len())
    } else {
        (values, 1)
    };
    Ok((
        json!({"calibration":model.decode(&x)?,"margin":model.margin(&x)?,"rows":rows,"cols":cols,"count":values.len()}),
        values,
    ))
}

pub(super) fn difference_step(request: &Value) -> Result<f64, String> {
    let step = match request.get("difference_step") {
        Some(value) => number(value)?,
        None => 0.002,
    };
    if !(0.000002..=0.01).contains(&step) {
        return Err("invalid numerical difference step".into());
    }
    Ok(step)
}
