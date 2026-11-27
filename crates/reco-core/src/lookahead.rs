//! Checked storage accounting for lookahead scheduling.

#[cfg(any(
    all(feature = "gpu", any(target_os = "linux", target_os = "windows")),
    test
))]
pub(crate) fn pool_slots(frames: usize) -> Result<usize, String> {
    // Immediate D3D11 submission uses two alternating slots. Buffered runs
    // retain the frame being presented plus exactly the requested future frames.
    if frames == 0 {
        return Ok(2);
    }
    frames
        .checked_add(1)
        .ok_or_else(|| "lookahead pool capacity overflow".into())
}

#[cfg(any(feature = "gpu", test))]
pub(crate) fn image_bytes(sizes: &[(u32, u32)], bytes_per_sample: usize) -> Result<usize, String> {
    sizes.iter().try_fold(0usize, |total, &(width, height)| {
        let y = (width as usize).checked_mul(height as usize);
        let chroma = (width as usize / 2)
            .checked_mul(height as usize / 2)
            .and_then(|n| n.checked_mul(2));
        y.zip(chroma)
            .and_then(|(y, uv)| y.checked_add(uv))
            .and_then(|n| n.checked_mul(bytes_per_sample))
            .and_then(|n| total.checked_add(n))
            .ok_or_else(|| "lookahead image byte size overflow".into())
    })
}

#[cfg(any(
    all(feature = "gpu", any(target_os = "linux", target_os = "windows")),
    test
))]
pub(crate) fn max_frames(per_slot: usize, budget: usize) -> usize {
    if per_slot == 0 {
        return 0;
    }
    (budget / per_slot).saturating_sub(1)
}

/// Tracks owned image allocations in the lookahead buffer.
pub(crate) struct ImageBudget {
    retained: usize,
    limit: usize,
}

impl ImageBudget {
    pub(crate) fn new(limit: usize) -> Result<Self, String> {
        if limit == 0 {
            return Err("enabled lookahead requires a nonzero image-memory budget".into());
        }
        Ok(Self { retained: 0, limit })
    }

    pub(crate) fn retain(&mut self, bytes: usize) -> Result<(), String> {
        let required = self
            .retained
            .checked_add(bytes)
            .ok_or_else(|| "lookahead retained byte size overflow".to_string())?;
        if required > self.limit {
            return Err(format!(
                "lookahead image storage needs {required} bytes, exceeding its {} byte budget; reduce lookahead or source resolution, or explicitly increase the image-memory budget",
                self.limit
            ));
        }
        self.retained = required;
        Ok(())
    }

    pub(crate) fn release(&mut self, bytes: usize) {
        self.retained = self
            .retained
            .checked_sub(bytes)
            .expect("released unaccounted lookahead bytes");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_sizes_and_exact_fit() {
        assert_eq!(pool_slots(0).unwrap(), 2);
        assert_eq!(pool_slots(1).unwrap(), 2);
        assert!(pool_slots(usize::MAX).is_err());
        assert!(image_bytes(&[(u32::MAX, u32::MAX)], 2).is_err());
        assert_eq!(
            image_bytes(&[(128, 96), (256, 192)], 1).unwrap(),
            (128 * 96 + 256 * 192) * 3 / 2
        );
        assert_eq!(
            image_bytes(&[(128, 96), (256, 192)], 2).unwrap(),
            (128 * 96 + 256 * 192) * 3
        );
        for cameras in [1, 2, 3] {
            assert_eq!(
                image_bytes(&vec![(128, 96); cameras], 2).unwrap(),
                128 * 96 * 3 * cameras
            );
        }
        for n in 1..100 {
            let bytes = pool_slots(n).unwrap() * 7;
            assert_eq!(max_frames(7, bytes), n);
            assert_eq!(max_frames(7, bytes - 1), n - 1);
        }
    }

    #[test]
    fn exact_limit_rejection_does_not_change_accounting() {
        assert!(ImageBudget::new(0).is_err());
        let mut b = ImageBudget::new(20).unwrap();
        b.retain(12).unwrap();
        assert!(b.retain(9).is_err());
        b.retain(8).unwrap();
        b.release(12);
        b.retain(12).unwrap();
        assert!(b.retain(usize::MAX).is_err());
        b.release(20);
        assert_eq!(b.retained, 0);
    }
}
