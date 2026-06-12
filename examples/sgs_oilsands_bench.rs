//! Native benchmark mirroring the variogram-demo SGSIM call.
//! Loads the same oilsands.dat, applies a representative spherical
//! variogram + anisotropy fit, runs one 77x138x7 SGS realization, and
//! prints wall time. Used to determine whether the multi-minute hang
//! in the browser is the WASM overhead or just the algorithm.
//!
//! Run with:
//!   OILSANDS_DAT=/path/to/oilsands.dat cargo run --release --example sgs_oilsands_bench
//!
//! Or pass the path as the first argument:
//!   cargo run --release --example sgs_oilsands_bench -- /path/to/oilsands.dat

use std::time::Instant;

use kriging_rs::{
    Anisotropy3D, Coord3D, Grid3D, GslibAnisotropy, PlanarDataset3D, Real, SgsModel3D,
    VariogramModel, VariogramType, from_gslib, gaussian_simulation_3d_stream,
};

fn main() {
    // Path source priority: first CLI arg, then OILSANDS_DAT env var.
    let path = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("OILSANDS_DAT").ok())
        .expect("supply oilsands.dat path as the first arg or via OILSANDS_DAT env var");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read oilsands.dat at {path}: {e}"));
    let mut lines = text.lines();
    let _title = lines.next().unwrap();
    let n_cols: usize = lines.next().unwrap().trim().parse().unwrap();
    for _ in 0..n_cols {
        let _ = lines.next();
    }
    let mut xs: Vec<Real> = Vec::new();
    let mut ys: Vec<Real> = Vec::new();
    let mut zs: Vec<Real> = Vec::new();
    let mut vs: Vec<Real> = Vec::new();
    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < 5 {
            continue;
        }
        let x: Real = cols[1].parse().unwrap();
        let y: Real = cols[2].parse().unwrap();
        let z: Real = cols[3].parse().unwrap();
        let v: Real = cols[4].parse().unwrap();
        if v <= -8.999 as Real {
            continue;
        }
        xs.push(x);
        ys.push(y);
        zs.push(z);
        vs.push(v);
    }
    println!("loaded {} samples", xs.len());

    // Fit parameters from the demo (azm=0, dip=0; anis1=183/174, anis2=75/174).
    let aniso: Anisotropy3D = from_gslib(GslibAnisotropy {
        ang1: 0.0,
        ang2: 0.0,
        ang3: 0.0,
        anis1: 183.0 / 174.0,
        anis2: 75.0 / 174.0,
    })
    .unwrap();
    let variogram = VariogramModel::new(
        15.9 as Real,
        25.8 as Real,
        174.0 as Real,
        VariogramType::Spherical,
    )
    .unwrap();

    let coords: Vec<Coord3D> = xs
        .iter()
        .zip(ys.iter())
        .zip(zs.iter())
        .map(|((&x, &y), &z)| Coord3D::new(x, y, z))
        .collect();
    let dataset = PlanarDataset3D::new(coords, vs).unwrap();
    let model = SgsModel3D::new(dataset, aniso, variogram).unwrap();

    // Match the demo defaults at the oilsands bbox:
    //   x: ~3330..~10005 (extent ~6675), dxdy = min(174,183)/4 = 43.5
    //     -> nx = ceil(6675/43.5) = 154
    //   y: similar 138 horizontal
    //   z: extent ~140, dz = 75/4 = 18.75 -> nz = ceil(140/18.75) = 8
    // The demo reported 77x138x7; that's because it used the actual
    // bbox extents which I'll let print the inputs of:
    let mut x_min: Real = Real::INFINITY;
    let mut x_max: Real = Real::NEG_INFINITY;
    let mut y_min: Real = Real::INFINITY;
    let mut y_max: Real = Real::NEG_INFINITY;
    let mut z_min: Real = Real::INFINITY;
    let mut z_max: Real = Real::NEG_INFINITY;
    for i in 0..xs.len() {
        if xs[i] < x_min {
            x_min = xs[i]
        }
        if xs[i] > x_max {
            x_max = xs[i]
        }
        if ys[i] < y_min {
            y_min = ys[i]
        }
        if ys[i] > y_max {
            y_max = ys[i]
        }
        if zs[i] < z_min {
            z_min = zs[i]
        }
        if zs[i] > z_max {
            z_max = zs[i]
        }
    }
    let dxdy: Real = 174.0 / 4.0;
    let dz: Real = 75.0 / 4.0;
    let nx = ((x_max - x_min) / dxdy).ceil().max(1.0) as usize;
    let ny = ((y_max - y_min) / dxdy).ceil().max(1.0) as usize;
    let nz = ((z_max - z_min) / dz).ceil().max(1.0) as usize;
    println!(
        "bbox: x [{:.0}..{:.0}], y [{:.0}..{:.0}], z [{:.0}..{:.0}]",
        x_min, x_max, y_min, y_max, z_min, z_max
    );
    println!("grid: {}x{}x{} = {} cells", nx, ny, nz, nx * ny * nz);

    let grid = Grid3D::new(
        nx,
        ny,
        nz,
        Coord3D::new(x_min + dxdy / 2.0, y_min + dxdy / 2.0, z_min + dz / 2.0),
        Coord3D::new(dxdy, dxdy, dz),
    )
    .unwrap();

    let t0 = Instant::now();
    let mut realisations_done = 0;
    gaussian_simulation_3d_stream(&model, &grid, 42, 1, |idx, gv| {
        realisations_done += 1;
        let elapsed = t0.elapsed();
        println!(
            "realization {} done in {:.2}s ({} cells)",
            idx,
            elapsed.as_secs_f64(),
            gv.len()
        );
        Ok::<(), kriging_rs::SgsError>(())
    })
    .unwrap();
    println!(
        "total wall: {:.2}s, {} cells/s",
        t0.elapsed().as_secs_f64(),
        (nx * ny * nz) as f64 / t0.elapsed().as_secs_f64()
    );
    println!("done. realizations: {}", realisations_done);
}
