//! GPU-accelerated image processing primitives for AKAZE calibration.
//!
//! Provides compute shader implementations of Gaussian blur, Scharr
//! derivatives, Perona-Malik conductivity, and FED nonlinear diffusion.
//! These are the operations that dominate AKAZE feature detection time
//! (99% of calibration wall clock).
//!
//! Uses wgpu compute shaders, working on all platforms that reco-core
//! supports (Vulkan, Metal, DX12).
//!
//! ## Usage
//!
//! ```ignore
//! let gpu_diff = GpuDiffusion::new(&gpu, &[(1920, 1440), (960, 720)]);
//! gpu_diff.upload_image(&gpu, 0, &grayscale_f32);
//! gpu_diff.evolve(&gpu, 0, 1.0, 0.7, &fed_steps);
//! let result = gpu_diff.readback_lt(&gpu, 0);
//! ```

use crate::gpu::GpuContext;
use wgpu::util::DeviceExt;

/// Workgroup size for 2D compute shaders (16x16 = 256 threads).
const WG_SIZE: u32 = 16;

/// Pre-allocated GPU buffers for one octave of the scale space.
struct OctaveBuffers {
    width: u32,
    height: u32,
    /// Ping-pong buffers for the diffused image (lt).
    lt_a: wgpu::Buffer,
    lt_b: wgpu::Buffer,
    /// Intermediate buffers for the evolution chain.
    lsmooth: wgpu::Buffer,
    lx: wgpu::Buffer,
    ly: wgpu::Buffer,
    lflow: wgpu::Buffer,
    /// Temporary buffer for separable filter intermediate result.
    temp: wgpu::Buffer,
    /// Staging buffer for CPU readback.
    staging: wgpu::Buffer,
}

/// GPU-accelerated nonlinear diffusion for AKAZE scale-space construction.
///
/// Compiles compute shaders once at creation, pre-allocates buffers per
/// octave, and provides methods to run the full evolution chain on GPU.
pub struct GpuDiffusion {
    // Pipelines
    separable_pipeline: wgpu::ComputePipeline,
    pm_g2_pipeline: wgpu::ComputePipeline,
    fed_pipeline: wgpu::ComputePipeline,

    // Bind group layouts
    separable_layout: wgpu::BindGroupLayout,
    pm_g2_layout: wgpu::BindGroupLayout,
    fed_layout: wgpu::BindGroupLayout,

    // Per-octave resources
    octaves: Vec<OctaveBuffers>,

    // Reusable uniform buffer (updated per dispatch)
    params_buffer: wgpu::Buffer,
}

impl GpuDiffusion {
    /// Create pipelines and allocate buffers for the given octave dimensions.
    ///
    /// `octave_dims` lists (width, height) for each octave level, largest first.
    /// Typically: `[(1920, 1440), (960, 720), (480, 360), (240, 180)]`.
    pub fn new(gpu: &GpuContext, octave_dims: &[(u32, u32)]) -> Self {
        let device = &gpu.device;

        // -- Compile shaders --
        let separable_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("separable_filter"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/separable_filter.wgsl").into()),
        });
        let pm_g2_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pm_g2"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/pm_g2.wgsl").into()),
        });
        let fed_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fed_diffusion"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fed_diffusion.wgsl").into()),
        });

        // -- Bind group layouts (must match shader bindings exactly) --
        // Separable filter: input(read) + output(rw) + params(uniform)
        let separable_layout = create_storage_rw_uniform_layout(device, "separable");
        // pm_g2: lx(read) + ly(read) + lflow(rw) + params(uniform)
        let pm_g2_layout = create_bind_group_layout(
            device,
            "pm_g2",
            &[true, true, false], // read, read, read_write
        );
        // FED: lt_in(read) + lt_out(rw) + lflow(read) + params(uniform)
        let fed_layout = create_bind_group_layout(
            device,
            "fed",
            &[true, false, true], // read, read_write, read
        );

        // -- Pipelines --
        let separable_pipeline =
            create_compute_pipeline(device, "separable", &separable_shader, &separable_layout);
        let pm_g2_pipeline = create_compute_pipeline(device, "pm_g2", &pm_g2_shader, &pm_g2_layout);
        let fed_pipeline = create_compute_pipeline(device, "fed", &fed_shader, &fed_layout);

        // -- Params uniform buffer (reused, updated per dispatch) --
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("diffusion_params"),
            size: 256, // enough for any of our param structs, aligned
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // -- Per-octave buffers --
        let octaves = octave_dims
            .iter()
            .map(|&(w, h)| create_octave_buffers(device, w, h))
            .collect();

        Self {
            separable_pipeline,
            pm_g2_pipeline,
            fed_pipeline,
            separable_layout,
            pm_g2_layout,
            fed_layout,
            octaves,
            params_buffer,
        }
    }

    /// Number of octave levels allocated.
    pub fn octave_count(&self) -> usize {
        self.octaves.len()
    }

    /// Upload f32 grayscale image data to the lt_a buffer for an octave.
    pub fn upload_image(&self, gpu: &GpuContext, octave: usize, data: &[f32]) {
        let oct = &self.octaves[octave];
        let bytes: &[u8] = bytemuck::cast_slice(data);
        gpu.queue.write_buffer(&oct.lt_a, 0, bytes);
    }

    /// Run the full evolution step on GPU: gaussian_blur -> scharr -> pm_g2 -> FED loop.
    ///
    /// After this call, the diffused image is in lt_a (or lt_b depending on
    /// the number of FED steps). Use `readback_lt()` to get the result.
    pub fn evolve(
        &self,
        gpu: &GpuContext,
        octave: usize,
        gaussian_sigma: f32,
        contrast_factor: f64,
        fed_tau_steps: &[f64],
    ) {
        let oct = &self.octaves[octave];
        let device = &gpu.device;
        let w = oct.width;
        let h = oct.height;
        let dispatch_x = w.div_ceil(WG_SIZE);
        let dispatch_y = h.div_ceil(WG_SIZE);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("diffusion_evolve"),
        });

        // Step 1: Gaussian blur lt_a -> lsmooth (separable: H then V)
        let gauss_kernel = gaussian_kernel(gaussian_sigma);
        let radius = (gauss_kernel.len() / 2) as u32;

        // Horizontal pass: lt_a -> temp
        self.dispatch_separable(
            &mut encoder,
            device,
            &oct.lt_a,
            &oct.temp,
            w,
            h,
            radius,
            0, // horizontal
            &gauss_kernel,
            dispatch_x,
            dispatch_y,
        );
        // Vertical pass: temp -> lsmooth
        self.dispatch_separable(
            &mut encoder,
            device,
            &oct.temp,
            &oct.lsmooth,
            w,
            h,
            radius,
            1, // vertical
            &gauss_kernel,
            dispatch_x,
            dispatch_y,
        );

        // Step 2: Scharr horizontal: lsmooth -> lx
        let scharr_h: [f32; 3] = [-1.0, 0.0, 1.0];
        self.dispatch_separable(
            &mut encoder,
            device,
            &oct.lsmooth,
            &oct.lx,
            w,
            h,
            1,
            0,
            &scharr_h,
            dispatch_x,
            dispatch_y,
        );

        // Step 3: Scharr vertical: lsmooth -> ly
        let scharr_v: [f32; 3] = [-1.0, 0.0, 1.0];
        self.dispatch_separable(
            &mut encoder,
            device,
            &oct.lsmooth,
            &oct.ly,
            w,
            h,
            1,
            1,
            &scharr_v,
            dispatch_x,
            dispatch_y,
        );

        // Step 4: pm_g2: (lx, ly) -> lflow
        let inv_k_sq = (1.0 / (contrast_factor * contrast_factor)) as f32;
        self.dispatch_pm_g2(
            &mut encoder,
            device,
            &oct.lx,
            &oct.ly,
            &oct.lflow,
            w,
            h,
            inv_k_sq,
            dispatch_x,
            dispatch_y,
        );

        // Step 5: FED diffusion loop (ping-pong lt_a <-> lt_b)
        // Pre-create all uniform buffers for FED steps to avoid per-dispatch allocation.
        let fed_param_buffers: Vec<wgpu::Buffer> = fed_tau_steps
            .iter()
            .map(|&tau| {
                let params = [w, h, (tau as f32).to_bits(), 0u32];
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("fed_params"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
            })
            .collect();

        // Encode all FED steps: pre-allocated param buffers, inline bind groups.
        for (step_idx, param_buf) in fed_param_buffers.iter().enumerate() {
            let (lt_in, lt_out) = if step_idx % 2 == 0 {
                (&oct.lt_a, &oct.lt_b)
            } else {
                (&oct.lt_b, &oct.lt_a)
            };

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.fed_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: lt_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: lt_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: oct.lflow.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: param_buf.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fed_pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        // After the loop, result is in lt_b if odd steps, lt_a if even.
        if fed_tau_steps.len() % 2 == 1 {
            let size = (w * h * 4) as u64;
            encoder.copy_buffer_to_buffer(&oct.lt_b, 0, &oct.lt_a, 0, size);
        }

        gpu.queue.submit(Some(encoder.finish()));
    }

    /// Read back the current lt (always in lt_a after evolve).
    pub fn readback_lt(&self, gpu: &GpuContext, octave: usize) -> Vec<f32> {
        let oct = &self.octaves[octave];
        let size = (oct.width * oct.height * 4) as u64;

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("diffusion_readback"),
            });
        encoder.copy_buffer_to_buffer(&oct.lt_a, 0, &oct.staging, 0, size);
        gpu.queue.submit(Some(encoder.finish()));

        let slice = oct.staging.slice(0..size);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        let mapped = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        oct.staging.unmap();

        result
    }

    // -- Internal dispatch helpers --

    fn dispatch_separable(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        width: u32,
        height: u32,
        radius: u32,
        direction: u32,
        kernel: &[f32],
        dispatch_x: u32,
        dispatch_y: u32,
    ) {
        // Build params: width, height, radius, direction + 4x vec4 kernel (16 floats, 16-byte aligned)
        // Layout: [u32 x4] [f32 x4] [f32 x4] [f32 x4] [f32 x4] = 80 bytes
        let mut params = [0u32; 4 + 16]; // 4 u32s + 16 f32s as u32 bits
        params[0] = width;
        params[1] = height;
        params[2] = radius;
        params[3] = direction;
        for (i, &k) in kernel.iter().enumerate() {
            if i < 16 {
                params[4 + i] = k.to_bits();
            }
        }
        let params_bytes: &[u8] = bytemuck::cast_slice(&params);
        self.write_params_and_dispatch(
            encoder,
            device,
            &self.separable_pipeline,
            &self.separable_layout,
            &[
                buffer_entry(
                    0,
                    input,
                    wgpu::BufferBindingType::Storage { read_only: true },
                ),
                buffer_entry(
                    1,
                    output,
                    wgpu::BufferBindingType::Storage { read_only: false },
                ),
                buffer_entry(2, &self.params_buffer, wgpu::BufferBindingType::Uniform),
            ],
            params_bytes,
            dispatch_x,
            dispatch_y,
        );
    }

    fn dispatch_pm_g2(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        lx: &wgpu::Buffer,
        ly: &wgpu::Buffer,
        lflow: &wgpu::Buffer,
        width: u32,
        height: u32,
        inv_k_sq: f32,
        dispatch_x: u32,
        dispatch_y: u32,
    ) {
        let params = [width, height, inv_k_sq.to_bits(), 0u32];
        let params_bytes: &[u8] = bytemuck::cast_slice(&params);
        self.write_params_and_dispatch(
            encoder,
            device,
            &self.pm_g2_pipeline,
            &self.pm_g2_layout,
            &[
                buffer_entry(0, lx, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(1, ly, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(
                    2,
                    lflow,
                    wgpu::BufferBindingType::Storage { read_only: false },
                ),
                buffer_entry(3, &self.params_buffer, wgpu::BufferBindingType::Uniform),
            ],
            params_bytes,
            dispatch_x,
            dispatch_y,
        );
    }

    fn dispatch_fed_step(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        lt_in: &wgpu::Buffer,
        lt_out: &wgpu::Buffer,
        lflow: &wgpu::Buffer,
        width: u32,
        height: u32,
        step_size: f32,
        dispatch_x: u32,
        dispatch_y: u32,
    ) {
        let params = [width, height, step_size.to_bits(), 0u32];
        let params_bytes: &[u8] = bytemuck::cast_slice(&params);
        self.write_params_and_dispatch(
            encoder,
            device,
            &self.fed_pipeline,
            &self.fed_layout,
            &[
                buffer_entry(
                    0,
                    lt_in,
                    wgpu::BufferBindingType::Storage { read_only: true },
                ),
                buffer_entry(
                    1,
                    lt_out,
                    wgpu::BufferBindingType::Storage { read_only: false },
                ),
                buffer_entry(
                    2,
                    lflow,
                    wgpu::BufferBindingType::Storage { read_only: true },
                ),
                buffer_entry(3, &self.params_buffer, wgpu::BufferBindingType::Uniform),
            ],
            params_bytes,
            dispatch_x,
            dispatch_y,
        );
    }

    fn write_params_and_dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        pipeline: &wgpu::ComputePipeline,
        layout: &wgpu::BindGroupLayout,
        entries: &[wgpu::BindGroupEntry<'_>],
        params_bytes: &[u8],
        dispatch_x: u32,
        dispatch_y: u32,
    ) {
        // Write params to the shared uniform buffer
        // Note: this is a queue operation that happens before the encoder's commands
        // We need to use encoder.copy_buffer_to_buffer or write directly
        // Actually, queue.write_buffer is fine as long as we submit the encoder after
        // But we're building the encoder incrementally... We need a separate approach.
        //
        // Solution: use a staging approach - write params into a CPU-mapped buffer
        // and copy. Or simpler: create a small buffer per dispatch.
        // For simplicity in this first pass, create a temp uniform buffer per dispatch.
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dispatch_params"),
            contents: params_bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Replace the params buffer entry with our temp buffer
        let mut final_entries: Vec<wgpu::BindGroupEntry<'_>> = entries.to_vec();
        for entry in &mut final_entries {
            if let wgpu::BindingResource::Buffer(ref mut buf_binding) = entry.resource {
                if std::ptr::eq(buf_binding.buffer, &self.params_buffer) {
                    buf_binding.buffer = &params_buf;
                }
            }
        }

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &final_entries,
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn create_storage_rw_uniform_layout(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            storage_entry(0, true),  // input (read)
            storage_entry(1, false), // output (read_write)
            uniform_entry(2),        // params
        ],
    })
}

/// Create a bind group layout with N storage buffers (read/write configurable) + 1 uniform.
fn create_bind_group_layout(
    device: &wgpu::Device,
    label: &str,
    read_only: &[bool],
) -> wgpu::BindGroupLayout {
    let mut entries: Vec<wgpu::BindGroupLayoutEntry> = read_only
        .iter()
        .enumerate()
        .map(|(i, &ro)| storage_entry(i as u32, ro))
        .collect();
    entries.push(uniform_entry(read_only.len() as u32));
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn buffer_entry(
    binding: u32,
    buffer: &wgpu::Buffer,
    ty: wgpu::BufferBindingType,
) -> wgpu::BindGroupEntry<'_> {
    let _ = ty; // used for documentation only in this helper
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}_layout")),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&format!("{label}_pipeline")),
        layout: Some(&pipeline_layout),
        module: shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn create_octave_buffers(device: &wgpu::Device, width: u32, height: u32) -> OctaveBuffers {
    let pixel_count = (width * height) as u64;
    let buf_size = pixel_count * 4; // f32 = 4 bytes

    let create_buf = |label: &str, usage: wgpu::BufferUsages| -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: buf_size,
            usage,
            mapped_at_creation: false,
        })
    };

    let storage_rw =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;

    OctaveBuffers {
        width,
        height,
        lt_a: create_buf("lt_a", storage_rw),
        lt_b: create_buf("lt_b", storage_rw),
        lsmooth: create_buf("lsmooth", storage_rw),
        lx: create_buf("lx", storage_rw),
        ly: create_buf("ly", storage_rw),
        lflow: create_buf("lflow", storage_rw),
        temp: create_buf("temp", storage_rw),
        staging: create_buf(
            "staging",
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        ),
    }
}

/// Compute a 1D Gaussian kernel for the given sigma.
fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let radius = (sigma * 2.0).ceil() as i32;
    let size = (2 * radius + 1) as usize;
    let mut kernel = vec![0.0f32; size];
    let mut sum = 0.0f32;
    let inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);

    for i in 0..size {
        let x = (i as i32 - radius) as f32;
        let val = (-x * x * inv_2sigma2).exp();
        kernel[i] = val;
        sum += val;
    }
    // Normalize
    for v in &mut kernel {
        *v /= sum;
    }
    kernel
}
