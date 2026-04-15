use std::f64::consts::PI;

#[allow(non_snake_case)]
pub fn fed_tau_by_process_time(T: f64, M: i32, tau_max: f64, reordering: bool) -> Vec<f64> {
    fed_tau_by_cycle_time(T / f64::from(M), tau_max, reordering)
}

fn fed_tau_by_cycle_time(t: f64, tau_max: f64, reordering: bool) -> Vec<f64> {
    let n = (f64::ceil(f64::sqrt(3.0 * t / tau_max + 0.25) - 0.5f64 - 1.0e-8) + 0.5) as usize;
    let scale = 3.0 * t / (tau_max * ((n * (n + 1)) as f64));
    fed_tau_internal(n, scale, tau_max, reordering)
}

fn fed_tau_internal(n: usize, scale: f64, tau_max: f64, reordering: bool) -> Vec<f64> {
    let tau: Vec<f64> = (0..n)
        .map(|k| {
            let c: f64 = 1.0f64 / (4.0f64 * (n as f64) + 2.0f64);
            let d: f64 = scale * tau_max / 2.0f64;
            let h = f64::cos(PI * (2.0f64 * (k as f64) + 1.0f64) * c);
            d / (h * h)
        })
        .collect();
    if reordering {
        let kappa = n / 2;
        let mut prime = n + 1;
        while !primal::is_prime(prime as u64) {
            prime += 1;
        }
        let mut k = 0;
        (0..n)
            .map(move |_| {
                let mut index = ((k + 1) * kappa) % prime - 1;
                while index >= n {
                    k += 1;
                    index = ((k + 1) * kappa) % prime - 1;
                }
                k += 1;
                tau[index]
            })
            .collect()
    } else {
        tau
    }
}
