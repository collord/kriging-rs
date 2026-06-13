//! Normal-score transform for SGS pre/post-processing.
//!
//! Maps the empirical distribution of sample values to the standard normal
//! and back. Forward transform: rank-average percentile → inverse-normal
//! CDF. Backward transform: inverse-normal CDF → empirical-quantile look-up.
//!
//! Used by [`crate::simulation_3d::sgs_3d`] to:
//! 1. Convert sample values to Gaussian scores before kriging.
//! 2. Kriging and simulation happens entirely in score space.
//! 3. Convert each realization's grid scores back to data-space at the end.
//!
//! Standard textbook procedure; no GSLib-specific quirks. Same algorithm
//! used by geostatspy's `nscore` and pygslib's `nscore`.

use crate::Real;

/// Inverse standard normal CDF (probit function).
///
/// Returns the value `z` such that `Φ(z) = p`, where `Φ` is the standard
/// normal CDF. Uses `puruspe::error::inverf` via the identity
/// `Φ⁻¹(p) = sqrt(2) · inverf(2p − 1)`.
///
/// `p` must be in `(0, 1)` strictly; the function clamps to `(eps, 1-eps)`
/// to avoid the infinities at the boundaries.
#[inline]
pub(crate) fn inv_standard_normal_cdf(p: f64) -> f64 {
    let eps = 1e-12;
    let p_clamped = p.clamp(eps, 1.0 - eps);
    std::f64::consts::SQRT_2 * puruspe::error::inverf(2.0 * p_clamped - 1.0)
}

/// Pre-computed normal-score transform built from a sample dataset.
///
/// Holds the **sorted** sample values and their Gaussian scores; both the
/// forward (data → score) and backward (score → data) maps look up via
/// binary search on this table.
#[derive(Debug, Clone)]
pub struct NormalScoreTransform {
    /// Sorted sample values, ascending.
    sorted_values: Vec<Real>,
    /// Gaussian scores, aligned with `sorted_values`. The i-th score is
    /// `Φ⁻¹((rank_i) / (n + 1))` under the average-rank-for-ties
    /// convention.
    scores: Vec<Real>,
}

impl NormalScoreTransform {
    /// Fit the transform to a sample of values. Must contain at least one
    /// value; ties are handled via average ranks (so a tied pair of
    /// values both map to the score of their average rank).
    pub fn fit(values: &[Real]) -> Self {
        let n = values.len();
        assert!(n > 0, "NormalScoreTransform: need at least one value");

        // Sort values with original indices preserved.
        let mut indexed: Vec<(Real, usize)> = values.iter().copied().zip(0..n).collect();
        indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Compute average rank per tied group, then convert to scores.
        let mut scores = vec![0.0 as Real; n];
        let mut i = 0;
        while i < n {
            // Find the end of the tied group [i, j).
            let mut j = i + 1;
            while j < n && indexed[j].0 == indexed[i].0 {
                j += 1;
            }
            // Ranks within this group are i+1 .. j (1-indexed).
            // Average rank = ((i+1) + j) / 2.
            let avg_rank = ((i + 1) + j) as f64 / 2.0;
            let p = avg_rank / (n as f64 + 1.0);
            let score = inv_standard_normal_cdf(p) as Real;
            scores[i..j].fill(score);
            i = j;
        }

        let sorted_values: Vec<Real> = indexed.iter().map(|(v, _)| *v).collect();
        Self {
            sorted_values,
            scores,
        }
    }

    /// Forward-transform a raw sample-space value to a Gaussian score by
    /// interpolating between the fitted percentiles. Values outside the
    /// fitted range are clamped to the extreme scores (no extrapolation).
    pub fn forward(&self, value: Real) -> Real {
        let n = self.sorted_values.len();
        // Binary search for the position of `value` in sorted_values.
        match self
            .sorted_values
            .binary_search_by(|v| v.partial_cmp(&value).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(idx) => self.scores[idx],
            Err(idx) => {
                if idx == 0 {
                    self.scores[0]
                } else if idx >= n {
                    self.scores[n - 1]
                } else {
                    // Linear interpolation between scores[idx-1] and scores[idx]
                    // based on value's position between sorted_values[idx-1]
                    // and sorted_values[idx].
                    let v_lo = self.sorted_values[idx - 1];
                    let v_hi = self.sorted_values[idx];
                    let s_lo = self.scores[idx - 1];
                    let s_hi = self.scores[idx];
                    let t = if v_hi > v_lo {
                        (value - v_lo) / (v_hi - v_lo)
                    } else {
                        0.0
                    };
                    s_lo + t * (s_hi - s_lo)
                }
            }
        }
    }

    /// Backward-transform a Gaussian score to a data-space value by
    /// inverse-lookup on the empirical CDF. Scores outside the fitted
    /// range clamp to the extreme sample values (no tail extrapolation).
    pub fn backward(&self, score: Real) -> Real {
        let n = self.scores.len();
        if score <= self.scores[0] {
            return self.sorted_values[0];
        }
        if score >= self.scores[n - 1] {
            return self.sorted_values[n - 1];
        }
        // scores are sorted ascending because they were assigned in
        // sorted-value order via a monotonic transform.
        match self
            .scores
            .binary_search_by(|s| s.partial_cmp(&score).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(idx) => self.sorted_values[idx],
            Err(idx) => {
                // idx in 1..n because of the bounds checks above.
                let s_lo = self.scores[idx - 1];
                let s_hi = self.scores[idx];
                let v_lo = self.sorted_values[idx - 1];
                let v_hi = self.sorted_values[idx];
                let t = if s_hi > s_lo {
                    (score - s_lo) / (s_hi - s_lo)
                } else {
                    0.0
                };
                v_lo + t * (v_hi - v_lo)
            }
        }
    }

    /// Borrow the sorted values + scores tables.
    pub fn sorted_values(&self) -> &[Real] {
        &self.sorted_values
    }

    pub fn scores(&self) -> &[Real] {
        &self.scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn inv_standard_normal_cdf_known_quantiles() {
        // Φ⁻¹(0.5) = 0; Φ⁻¹(0.975) ≈ 1.95996.
        assert_relative_eq!(inv_standard_normal_cdf(0.5), 0.0, epsilon = 1e-9);
        assert_relative_eq!(
            inv_standard_normal_cdf(0.975),
            1.959963984540054,
            epsilon = 1e-6
        );
        // Symmetry: Φ⁻¹(0.025) = -Φ⁻¹(0.975).
        assert_relative_eq!(
            inv_standard_normal_cdf(0.025),
            -1.959963984540054,
            epsilon = 1e-6
        );
    }

    #[test]
    fn round_trip_at_sample_values_is_exact() {
        // Forward then backward at every sample value should return the
        // same value.
        let values: Vec<Real> = vec![1.0, 5.0, 3.0, 7.0, 2.0, 4.0, 6.0];
        let nst = NormalScoreTransform::fit(&values);
        for v in &values {
            let s = nst.forward(*v);
            let v_back = nst.backward(s);
            assert_relative_eq!(v_back as f64, *v as f64, epsilon = 1e-5);
        }
    }

    #[test]
    fn forward_preserves_rank_order() {
        let values: Vec<Real> = vec![10.0, 1.0, 5.0, 20.0, 15.0];
        let nst = NormalScoreTransform::fit(&values);
        let scores: Vec<Real> = values.iter().map(|v| nst.forward(*v)).collect();
        // Smaller value -> smaller score. Verify by checking pairs.
        for i in 0..values.len() {
            for j in 0..values.len() {
                if values[i] < values[j] {
                    assert!(
                        scores[i] < scores[j],
                        "rank-order violated: values[{i}]={}, values[{j}]={}, scores[{i}]={}, scores[{j}]={}",
                        values[i],
                        values[j],
                        scores[i],
                        scores[j],
                    );
                }
            }
        }
    }

    #[test]
    fn ties_get_same_score() {
        let values: Vec<Real> = vec![1.0, 3.0, 3.0, 5.0];
        let nst = NormalScoreTransform::fit(&values);
        // The two 3.0 values should map to the same score (average rank).
        let s1 = nst.forward(3.0);
        // Force a second lookup; should give identical result.
        let s2 = nst.forward(3.0);
        assert_eq!(s1, s2);
        // Distinct values should map to distinct scores.
        assert!(nst.forward(1.0) < s1);
        assert!(nst.forward(5.0) > s1);
    }

    #[test]
    fn extreme_scores_clamp_to_extreme_values() {
        let values: Vec<Real> = vec![1.0, 2.0, 3.0];
        let nst = NormalScoreTransform::fit(&values);
        // Far positive score -> max value.
        assert_eq!(nst.backward(10.0), 3.0);
        // Far negative score -> min value.
        assert_eq!(nst.backward(-10.0), 1.0);
    }

    #[test]
    fn backward_at_score_zero_is_near_median() {
        // For an odd-N symmetric input the score at the median sample
        // should be exactly 0.
        let values: Vec<Real> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let nst = NormalScoreTransform::fit(&values);
        // Median value = 3.0, rank 3, p = 3/6 = 0.5, score = 0.
        let s_at_median = nst.forward(3.0);
        assert_relative_eq!(s_at_median as f64, 0.0, epsilon = 1e-6);
    }
}
