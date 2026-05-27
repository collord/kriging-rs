//! Regular 3-D simulation grid.
//!
//! Defines the discretization of space at which SGS will draw simulated
//! values. Each cell has a linear index `i + nx·j + nx·ny·k` for cell
//! `(i, j, k)`, and a center coordinate `(origin + i·spacing.x, ...)`.

use crate::Real;
use crate::coord_3d::Coord3D;
use crate::error::KrigingError;

/// A regular 3-D grid: `nx × ny × nz` cells starting at `origin` with
/// constant `spacing` per dimension.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid3D {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub origin: Coord3D,
    pub spacing: Coord3D,
}

impl Grid3D {
    /// Construct a grid. Dimensions must be > 0 and spacing components
    /// must be strictly positive.
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        origin: Coord3D,
        spacing: Coord3D,
    ) -> Result<Self, KrigingError> {
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(KrigingError::InvalidInput(format!(
                "grid dimensions must be > 0: nx={nx}, ny={ny}, nz={nz}"
            )));
        }
        if !spacing.x.is_finite()
            || !spacing.y.is_finite()
            || !spacing.z.is_finite()
            || spacing.x <= 0.0
            || spacing.y <= 0.0
            || spacing.z <= 0.0
        {
            return Err(KrigingError::InvalidInput(format!(
                "grid spacing must be finite and positive: ({}, {}, {})",
                spacing.x, spacing.y, spacing.z
            )));
        }
        Ok(Self {
            nx,
            ny,
            nz,
            origin,
            spacing,
        })
    }

    /// Total number of cells in the grid.
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Linear index for cell `(i, j, k)`. Row-major along x first, then y,
    /// then z (the usual "i + nx*j + nx*ny*k" convention).
    #[inline]
    pub fn linear_index(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + self.ny * k)
    }

    /// Inverse of [`linear_index`]: `(i, j, k)` for a given linear index.
    #[inline]
    pub fn ijk(&self, linear: usize) -> (usize, usize, usize) {
        let i = linear % self.nx;
        let rest = linear / self.nx;
        let j = rest % self.ny;
        let k = rest / self.ny;
        (i, j, k)
    }

    /// World-space center coordinate of cell `(i, j, k)`.
    #[inline]
    pub fn cell_center(&self, i: usize, j: usize, k: usize) -> Coord3D {
        Coord3D::new(
            self.origin.x + (i as Real) * self.spacing.x,
            self.origin.y + (j as Real) * self.spacing.y,
            self.origin.z + (k as Real) * self.spacing.z,
        )
    }

    /// World-space center of the cell at the given linear index.
    #[inline]
    pub fn cell_center_linear(&self, linear: usize) -> Coord3D {
        let (i, j, k) = self.ijk(linear);
        self.cell_center(i, j, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_grid() -> Grid3D {
        Grid3D::new(
            4,
            3,
            2,
            Coord3D::new(0.0, 0.0, 0.0),
            Coord3D::new(1.0, 1.0, 1.0),
        )
        .unwrap()
    }

    #[test]
    fn rejects_zero_dimension() {
        assert!(matches!(
            Grid3D::new(
                0,
                1,
                1,
                Coord3D::new(0.0, 0.0, 0.0),
                Coord3D::new(1.0, 1.0, 1.0)
            ),
            Err(KrigingError::InvalidInput(_))
        ));
    }

    #[test]
    fn rejects_nonpositive_spacing() {
        assert!(matches!(
            Grid3D::new(
                2,
                2,
                2,
                Coord3D::new(0.0, 0.0, 0.0),
                Coord3D::new(0.0, 1.0, 1.0)
            ),
            Err(KrigingError::InvalidInput(_))
        ));
    }

    #[test]
    fn n_cells_matches_product() {
        let g = unit_grid();
        assert_eq!(g.n_cells(), 24);
    }

    #[test]
    fn linear_index_and_ijk_are_inverses() {
        let g = unit_grid();
        for linear in 0..g.n_cells() {
            let (i, j, k) = g.ijk(linear);
            assert!(i < g.nx && j < g.ny && k < g.nz);
            assert_eq!(g.linear_index(i, j, k), linear);
        }
    }

    #[test]
    fn cell_centers_are_evenly_spaced() {
        let g = Grid3D::new(
            3,
            1,
            1,
            Coord3D::new(0.0, 0.0, 0.0),
            Coord3D::new(2.0, 1.0, 1.0),
        )
        .unwrap();
        assert_eq!(g.cell_center(0, 0, 0), Coord3D::new(0.0, 0.0, 0.0));
        assert_eq!(g.cell_center(1, 0, 0), Coord3D::new(2.0, 0.0, 0.0));
        assert_eq!(g.cell_center(2, 0, 0), Coord3D::new(4.0, 0.0, 0.0));
    }
}
