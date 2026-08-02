/**
 * Central registry of help content shown by the `?` {@link HelpTip} icons across the demo.
 *
 * Keeping every explanation in one module means the copy has a single source of truth and can
 * be reviewed for accuracy independently of the UI wiring. Each entry has a short `title` and a
 * `content` node (kept to a few sentences so it fits comfortably in a popover).
 */

const HELP_TOPICS = {
  // ---- Section overviews -------------------------------------------------
  dataSection: {
    title: "Data",
    content: (
      <p>
        Bring your own points as a CSV, or leave it empty to interpolate a built-in synthetic
        dataset. Uploaded data feeds the 2D Surface demo below.
      </p>
    ),
  },
  surfaceSection: {
    title: "2D Surface demo",
    content: (
      <>
        <p>
          Fits a variogram to your samples, builds a kriging model, and predicts a value at every
          cell of a regular grid — then draws the result as a heatmap.
        </p>
        <p>
          The variogram plot and residual plot below help you judge whether the fit is
          reasonable before trusting the surface.
        </p>
      </>
    ),
  },
  compareSection: {
    title: "Compare variogram models",
    content: (
      <p>
        Krige the same synthetic dataset with two variogram families side by side. Use it to see
        how the model choice reshapes the surface — differences are usually largest away from the
        data points.
      </p>
    ),
  },

  // ---- Kriging / grid controls ------------------------------------------
  krigingMode: {
    title: "Kriging mode",
    content: (
      <>
        <p>
          <strong>Ordinary</strong> interpolates a continuous quantity (e.g. temperature) with an
          unknown but locally constant mean.
        </p>
        <p>
          <strong>Binomial</strong> models a proportion from <code>successes</code> /{" "}
          <code>trials</code> on the logit scale and returns a prevalence in (0, 1). Choose the one
          that matches your data columns.
        </p>
      </>
    ),
  },
  gridResolution: {
    title: "Grid resolution",
    content: (
      <p>
        Number of cells per side of the prediction grid. Higher resolution gives a smoother, more
        detailed heatmap but predicts more points (48×48 = 2,304 predictions), so it takes longer.
      </p>
    ),
  },
  empiricalBins: {
    title: "Empirical bins",
    content: (
      <p>
        Point pairs are grouped into this many distance bins to form the empirical variogram that
        the model is fitted to. Fewer bins are smoother but coarser; more bins capture finer
        structure but are noisier. 12 is a sensible default.
      </p>
    ),
  },
  maxDistance: {
    title: "Max distance",
    content: (
      <p>
        Pairs farther apart than this are ignored when building the empirical variogram. Leave it
        blank to let the library choose automatically (about half the data’s extent). Lowering it
        focuses the fit on short-range structure.
      </p>
    ),
  },
  surfaceLayer: {
    title: "Surface layer",
    content: (
      <>
        <p>
          <strong>Prediction</strong> shows the kriged estimate at each cell.
        </p>
        <p>
          <strong>Kriging variance</strong> shows the model’s uncertainty instead — low near data
          points, higher far from them. It’s a map of confidence rather than value.
        </p>
      </>
    ),
  },
  binomialPrior: {
    title: "Binomial prior (α, β)",
    content: (
      <>
        <p>
          A <code>Beta(α, β)</code> prior used only in Binomial mode. It shrinks proportions from
          low-trial sites toward a baseline before interpolation, stabilizing noisy estimates.
        </p>
        <p>
          <code>Beta(0.5, 0.5)</code> is a weak (Jeffreys) prior; <code>Beta(1, 1)</code> is
          uniform. Larger values pull harder toward the prior mean.
        </p>
      </>
    ),
  },
  residualPlot: {
    title: "Residual plot",
    content: (
      <>
        <p>
          Residuals are observed − predicted at the sample locations. This picks how to view them:
        </p>
        <ul>
          <li>
            <strong>Scatter</strong> — each residual individually.
          </li>
          <li>
            <strong>Histogram</strong> — their distribution.
          </li>
          <li>
            <strong>QQ-style</strong> — a check for normality.
          </li>
        </ul>
        <p>Residuals centered on zero with no visible pattern indicate a good fit.</p>
      </>
    ),
  },
  backend: {
    title: "Compute backend",
    content: (
      <p>
        Where the kriging solve runs. <strong>Auto</strong> uses WebGPU when the browser supports
        it and falls back to CPU otherwise. WebGPU can accelerate large prediction batches; CPU is
        always available.
      </p>
    ),
  },
  performanceHarness: {
    title: "Performance harness",
    content: (
      <p>
        Runs a fixed benchmark (350 samples → 36×36 grid, several warm-up and measured iterations)
        and reports timing broken down by phase: data prep, variogram fit, model build, batch
        prediction, and result mapping.
      </p>
    ),
  },

  // ---- Result read-outs --------------------------------------------------
  empiricalVariogramPlot: {
    title: "Empirical variogram",
    content: (
      <p>
        Dots are the empirical semivariance in each distance bin. A rise from a low{" "}
        <em>nugget</em> up to a <em>sill</em> plateau, reached at the <em>range</em>, is the
        classic pattern: nearby points are more alike, and beyond the range there’s no more
        spatial correlation.
      </p>
    ),
  },
  residualStats: {
    title: "Residual statistics",
    content: (
      <p>
        <strong>Mean</strong> near 0 means little systematic bias. <strong>RMSE</strong>{" "}
        (root-mean-square error) summarizes the typical prediction error at the sample locations,
        in the data’s own units — smaller is better.
      </p>
    ),
  },
  surfaceRange: {
    title: "Value / variance range",
    content: (
      <p>
        The minimum and maximum of the currently displayed surface layer, which set the color
        scale of the heatmap.
      </p>
    ),
  },

  // ---- Data / environment ------------------------------------------------
  csvFormat: {
    title: "CSV format",
    content: (
      <>
        <p>Headers are case-insensitive. Required columns:</p>
        <ul>
          <li>
            <code>lat</code>, <code>lon</code> — always.
          </li>
          <li>
            <code>value</code> — for ordinary kriging, <em>or</em>
          </li>
          <li>
            <code>successes</code> + <code>trials</code> — for binomial kriging.
          </li>
        </ul>
        <p>Rows with non-numeric lat/lon are skipped; at least 3 valid rows are required.</p>
      </>
    ),
  },
  webgpu: {
    title: "WebGPU status",
    content: (
      <p>
        Whether this browser exposes WebGPU, an API for running compute on the GPU. When
        available, the <strong>Auto</strong> backend uses it to speed up large prediction batches;
        otherwise everything runs on the CPU.
      </p>
    ),
  },
};

export default HELP_TOPICS;
