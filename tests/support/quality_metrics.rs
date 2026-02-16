use anyhow::{bail, Result};

/// Quality metrics for reconstructed RGB data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityMetrics {
    pub mae: f64,
    pub mse: f64,
    pub psnr: f64,
    pub reproduction_rate: f64,
}

/// Computes `MAE`, `MSE`, `PSNR`, and `reproduction_rate`.
pub fn calculate_quality_metrics(reference: &[u8], decoded: &[u8]) -> Result<QualityMetrics> {
    if reference.len() != decoded.len() {
        bail!(
            "input length mismatch: reference={}, decoded={}",
            reference.len(),
            decoded.len(),
        );
    }
    if reference.is_empty() {
        bail!("input must not be empty");
    }

    let mut abs_sum = 0_f64;
    let mut sq_sum = 0_f64;
    for (expected, actual) in reference.iter().zip(decoded.iter()) {
        let diff = f64::from(*expected) - f64::from(*actual);
        abs_sum += diff.abs();
        sq_sum += diff * diff;
    }

    let n = reference.len() as f64;
    let mae = abs_sum / n;
    let mse = sq_sum / n;
    let psnr = if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * ((255.0_f64 * 255.0_f64) / mse).log10()
    };
    let reproduction_rate = 1.0 - (mae / 255.0);

    Ok(QualityMetrics {
        mae,
        mse,
        psnr,
        reproduction_rate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_buffers_have_perfect_metrics() {
        let reference = [0_u8, 10, 100, 255];
        let metrics = calculate_quality_metrics(&reference, &reference).unwrap();

        assert_eq!(metrics.mae, 0.0);
        assert_eq!(metrics.mse, 0.0);
        assert!(metrics.psnr.is_infinite());
        assert_eq!(metrics.reproduction_rate, 1.0);
    }

    #[test]
    fn opposite_buffers_have_zero_reproduction_rate() {
        let reference = [0_u8, 0, 0, 0];
        let decoded = [255_u8, 255, 255, 255];
        let metrics = calculate_quality_metrics(&reference, &decoded).unwrap();

        assert_eq!(metrics.mae, 255.0);
        assert_eq!(metrics.reproduction_rate, 0.0);
    }
}
