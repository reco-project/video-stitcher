use super::image::GrayFloatImage;
use super::{Akaze, fed_tau};
use log::*;

#[derive(Debug)]
#[allow(non_snake_case)]
pub struct EvolutionStep {
    pub etime: f64,
    pub esigma: f64,
    pub octave: u32,
    pub sublevel: u32,
    pub sigma_size: u32,
    pub lt: GrayFloatImage,
    pub lsmooth: GrayFloatImage,
    pub lx: GrayFloatImage,
    pub ly: GrayFloatImage,
    pub lxx: GrayFloatImage,
    pub lyy: GrayFloatImage,
    pub lxy: GrayFloatImage,
    pub lflow: GrayFloatImage,
    pub ldet: GrayFloatImage,
    pub fed_tau_steps: Vec<f64>,
}

impl EvolutionStep {
    fn new(octave: u32, sublevel: u32, options: &Akaze) -> EvolutionStep {
        let esigma = options.base_scale_offset
            * f64::powf(
                2.0f64,
                f64::from(sublevel) / f64::from(options.num_sublevels) + f64::from(octave),
            );
        let etime = 0.5 * (esigma * esigma);
        EvolutionStep {
            etime,
            esigma,
            octave,
            sublevel,
            sigma_size: esigma.round() as u32,
            lt: GrayFloatImage::new(0, 0),
            lsmooth: GrayFloatImage::new(0, 0),
            lx: GrayFloatImage::new(0, 0),
            ly: GrayFloatImage::new(0, 0),
            lxx: GrayFloatImage::new(0, 0),
            lyy: GrayFloatImage::new(0, 0),
            lxy: GrayFloatImage::new(0, 0),
            lflow: GrayFloatImage::new(0, 0),
            ldet: GrayFloatImage::new(0, 0),
            fed_tau_steps: vec![],
        }
    }
}

impl Akaze {
    pub fn allocate_evolutions(&self, width: u32, height: u32) -> Vec<EvolutionStep> {
        let mut evolutions: Vec<EvolutionStep> = (0..self.max_octave_evolution)
            .filter_map(|octave| {
                let rfactor = 2.0f64.powi(-(octave as i32));
                let level_height = (f64::from(height) * rfactor) as u32;
                let level_width = (f64::from(width) * rfactor) as u32;
                let smallest_dim = std::cmp::min(level_width, level_height);
                if smallest_dim < 40 {
                    None
                } else {
                    let sublevels = if smallest_dim < 80 {
                        1
                    } else {
                        self.num_sublevels
                    };
                    Some(
                        (0..sublevels)
                            .map(move |sublevel| EvolutionStep::new(octave, sublevel, self)),
                    )
                }
            })
            .flatten()
            .collect();
        for i in 1..evolutions.len() {
            let ttime = evolutions[i].etime - evolutions[i - 1].etime;
            evolutions[i].fed_tau_steps = fed_tau::fed_tau_by_process_time(ttime, 1, 0.25, true);
            debug!(
                "{} steps in evolution {}.",
                evolutions[i].fed_tau_steps.len(),
                i
            );
        }
        evolutions
    }
}
