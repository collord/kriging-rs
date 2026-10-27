use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use futures_channel::oneshot;
use wgpu::util::DeviceExt;

use crate::Real;
use crate::distance::GeoCoord;
use crate::variogram::models::{VariogramModel, VariogramType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    WebGpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuSupport {
    pub available: bool,
    pub backend: GpuBackend,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VariogramGpuParams {
    n_train: u32,
    n_pred: u32,
    variogram_type: u32,
    _pad0: u32,
    nugget: f32,
    sill: f32,
    range: f32,
    shape: f32,
}

const COVARIANCE_SHADER: &str = r#"
struct Params {
    n_train: u32,
    n_pred: u32,
    variogram_type: u32,
    _pad0: u32,
    nugget: f32,
    sill: f32,
    range: f32,
    shape: f32,
};

@group(0) @binding(0)
var<storage, read> train_coords: array<vec2<f32>>;
@group(0) @binding(1)
var<storage, read> pred_coords: array<vec2<f32>>;
@group(0) @binding(2)
var<storage, read_write> out_covariances: array<f32>;
@group(0) @binding(3)
var<uniform> params: Params;

fn semivariance(distance: f32, variogram_type: u32, nugget: f32, sill: f32, range_in: f32, shape_in: f32) -> f32 {
    // gamma(0) = 0: the nugget is the jump just past the origin (matches VariogramModel).
    if distance <= 0.0 {
        return 0.0;
    }
    let d = max(distance, 0.0);
    let r = max(range_in, 1e-6);
    let partial = max(sill - nugget, 0.0);

    if variogram_type == 0u {
        if d >= r {
            return sill;
        }
        let x = d / r;
        return nugget + partial * (1.5 * x - 0.5 * x * x * x);
    }
    if variogram_type == 1u {
        return nugget + partial * (1.0 - exp(-3.0 * d / r));
    }
    if variogram_type == 2u {
        return nugget + partial * (1.0 - exp(-3.0 * d * d / (r * r)));
    }
    if variogram_type == 3u {
        if d >= r {
            return sill;
        }
        let x = d / r;
        let poly = 7.0 * x * x - 8.5 * x * x * x + 3.5 * pow(x, 5.0) - 0.5 * pow(x, 7.0);
        return nugget + partial * poly;
    }
    if variogram_type == 4u {
        let alpha = max(shape_in, 1e-6);
        let x = pow(d / r, alpha);
        return nugget + partial * (1.0 - exp(-x));
    }
    return nugget + partial * (1.0 - exp(-3.0 * d * d / (r * r)));
}

fn haversine_km(a: vec2<f32>, b: vec2<f32>) -> f32 {
    let deg_to_rad = 0.017453292519943295;
    let r_earth = 6371.0;
    let lat1 = a.x * deg_to_rad;
    let lon1 = a.y * deg_to_rad;
    let lat2 = b.x * deg_to_rad;
    let lon2 = b.y * deg_to_rad;

    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;

    let sdlat = sin(dlat * 0.5);
    let sdlon = sin(dlon * 0.5);
    let h = sdlat * sdlat + cos(lat1) * cos(lat2) * sdlon * sdlon;
    let c = 2.0 * atan2(sqrt(h), sqrt(max(1.0 - h, 0.0)));
    return r_earth * c;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let train_idx = gid.x;
    let pred_idx = gid.y;
    if train_idx >= params.n_train || pred_idx >= params.n_pred {
        return;
    }
    let d = haversine_km(train_coords[train_idx], pred_coords[pred_idx]);
    let semi = semivariance(d, params.variogram_type, params.nugget, params.sill, params.range, params.shape);
    let idx = pred_idx * params.n_train + train_idx;
    out_covariances[idx] = max(params.sill - semi, 0.0);
}
"#;

async fn request_adapter_best_effort(instance: &wgpu::Instance) -> Result<wgpu::Adapter, String> {
    let attempts = [
        wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        },
        wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            force_fallback_adapter: false,
            compatible_surface: None,
        },
        wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            force_fallback_adapter: false,
            compatible_surface: None,
        },
        wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            force_fallback_adapter: true,
            compatible_surface: None,
        },
    ];

    let mut last_err = None;
    for options in attempts {
        match instance.request_adapter(&options).await {
            Ok(adapter) => return Ok(adapter),
            Err(err) => last_err = Some(err.to_string()),
        }
    }
    Err(last_err.unwrap_or_else(|| "no adapter found".to_string()))
}

pub async fn detect_gpu_support() -> GpuSupport {
    let instance = wgpu::Instance::default();
    let adapter_result = request_adapter_best_effort(&instance).await;

    GpuSupport {
        available: adapter_result.is_ok(),
        backend: GpuBackend::WebGpu,
    }
}

fn encode_variogram_params(
    variogram: VariogramModel,
    n_train: usize,
    n_pred: usize,
) -> Result<VariogramGpuParams, String> {
    let (nugget, sill, range) = variogram.params();
    let (variogram_type, shape) = match variogram.variogram_type() {
        VariogramType::Spherical => (0u32, 0.0f32),
        VariogramType::Exponential => (1, 0.0),
        VariogramType::Gaussian => (2, 0.0),
        VariogramType::Cubic => (3, 0.0),
        VariogramType::Stable => (4, variogram.shape().unwrap_or(1.0)),
        VariogramType::Matern => {
            return Err("Matérn variogram is not supported on GPU; use CPU path".to_string());
        }
        VariogramType::Power => {
            return Err("Power variogram is not supported on GPU; use CPU path".to_string());
        }
        VariogramType::HoleEffect => {
            return Err("Hole-effect variogram is not supported on GPU; use CPU path".to_string());
        }
    };
    Ok(VariogramGpuParams {
        n_train: n_train as u32,
        n_pred: n_pred as u32,
        variogram_type,
        _pad0: 0,
        nugget,
        sill,
        range,
        shape,
    })
}

async fn read_buffer_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
) -> Result<Vec<Real>, String> {
    let (tx, rx) = oneshot::channel::<Result<Vec<Real>, String>>();
    wgpu::util::DownloadBuffer::read_buffer(device, queue, &buffer.slice(..), move |result| {
        let mapped = result
            .map_err(|e| format!("failed to map GPU output: {e}"))
            .and_then(|download| {
                let bytes: &[u8] = &download;
                if !bytes.len().is_multiple_of(std::mem::size_of::<Real>()) {
                    return Err("mapped GPU output had invalid byte length".to_string());
                }
                Ok(bytemuck::cast_slice::<u8, Real>(bytes).to_vec())
            });
        let _ = tx.send(mapped);
    });
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
    }
    rx.await
        .map_err(|_| "GPU readback channel was unexpectedly closed".to_string())?
}

pub async fn build_rhs_covariances_gpu(
    train_coords: &[GeoCoord],
    pred_coords: &[GeoCoord],
    variogram: VariogramModel,
) -> Result<Vec<Real>, String> {
    if train_coords.is_empty() || pred_coords.is_empty() {
        return Ok(Vec::new());
    }
    let total = train_coords
        .len()
        .checked_mul(pred_coords.len())
        .ok_or_else(|| "RHS covariance matrix size overflowed".to_string())?;

    let instance = wgpu::Instance::default();
    let adapter = request_adapter_best_effort(&instance)
        .await
        .map_err(|e| format!("no compatible GPU adapter found: {e}"))?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .map_err(|e| format!("failed to request GPU device: {e}"))?;

    let train_flat = train_coords
        .iter()
        .flat_map(|c| [c.lat(), c.lon()])
        .collect::<Vec<_>>();
    let pred_flat = pred_coords
        .iter()
        .flat_map(|c| [c.lat(), c.lon()])
        .collect::<Vec<_>>();
    let params = encode_variogram_params(variogram, train_coords.len(), pred_coords.len())?;

    let train_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("kriging_train_coords"),
        contents: bytemuck::cast_slice(&train_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let pred_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("kriging_pred_coords"),
        contents: bytemuck::cast_slice(&pred_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("kriging_rhs_covariances"),
        size: (total * std::mem::size_of::<Real>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("kriging_variogram_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("kriging_covariance_shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(COVARIANCE_SHADER)),
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("kriging_covariance_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("kriging_covariance_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("kriging_covariance_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("kriging_covariance_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: train_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: pred_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("kriging_covariance_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("kriging_covariance_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let groups_x = (train_coords.len() as u32).div_ceil(16);
        let groups_y = (pred_coords.len() as u32).div_ceil(16);
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }
    queue.submit(Some(encoder.finish()));

    read_buffer_f32(&device, &queue, &out_buffer).await
}

/// Smoke test: verifies a GPU adapter is available and then squares each value **on the CPU**.
///
/// The name is a legacy of earlier iterations where this helper ran an actual WGSL shader.
/// It is retained as an API-level probe for GPU availability with a trivial payload; do not
/// rely on it for performance. For real GPU work, use [`build_rhs_covariances_gpu`].
#[doc(alias = "gpu_probe_square")]
pub async fn gpu_square(values: &[Real]) -> Result<Vec<Real>, String> {
    let support = detect_gpu_support().await;
    if !support.available {
        return Err("no compatible GPU adapter found".to_string());
    }
    Ok(values.iter().map(|v| *v * *v).collect())
}

#[cfg(all(feature = "gpu-blocking", not(target_arch = "wasm32")))]
pub fn detect_gpu_support_blocking() -> GpuSupport {
    pollster::block_on(detect_gpu_support())
}

#[cfg(all(feature = "gpu-blocking", not(target_arch = "wasm32")))]
pub fn build_rhs_covariances_gpu_blocking(
    train_coords: &[GeoCoord],
    pred_coords: &[GeoCoord],
    variogram: VariogramModel,
) -> Result<Vec<Real>, String> {
    pollster::block_on(build_rhs_covariances_gpu(
        train_coords,
        pred_coords,
        variogram,
    ))
}
