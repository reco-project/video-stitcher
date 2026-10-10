//! Deterministic frame selection over ordered captures on one source clock.
//! The caller owns clock mapping, cadence, waiting, cancellation and image storage.

use std::collections::VecDeque;

#[derive(Clone, Copy, Debug)]
/// Explicit capture selection policy; no platform or cadence defaults are implied.
pub struct Config {
    /// Latest arrival permitted for a slot, measured from its scheduled time.
    pub wait_us: i64,
    /// Maximum spread among captures considered together for replacement; also limits captures ahead of the scheduled slot. An input with no eligible replacement may retain its previous image up to `max_age_us`.
    pub skew_us: i64,
    /// Maximum capture age at the selected set's media assessment time.
    pub max_age_us: i64,
    /// Pending images per input, excluding its one previously selected image.
    pub capacity: usize,
}

#[derive(Debug, PartialEq)]
/// Invalid policy, timeline or inability to produce an admissible complete set.
pub enum Error {
    /// Invalid dimensions or limits.
    Config,
    /// Missing ordering, negative time, inconsistent slot or timestamp overflow.
    Clock,
    /// An input has never supplied an eligible image.
    Missing(usize),
    /// An input's previous image is too old.
    Expired(usize),
    /// Available new images do not admit a bounded-skew set.
    Skew,
}

#[derive(Clone, Debug)]
/// Original capture time and its retained immutable image handle.
pub struct Capture<T> {
    /// Mapped source time, fixed for this capture's lifetime.
    pub time_us: i64,
    /// Caller-owned shared handle; selection never accesses pixel storage.
    pub image: T,
}

struct Queued<T> {
    capture: Capture<T>,
    arrived_us: i64,
}

#[derive(Debug)]
/// One image per input, preserving original capture identity.
pub struct Selection<T> {
    /// Media assessment time, also suitable for output PTS after origin adjustment.
    pub at_us: i64,
    /// Images in input order, including any held captures.
    pub captures: Vec<Capture<T>>,
}

/// Bounded pure state machine; the caller performs all waiting and I/O.
pub struct Selector<T> {
    config: Config,
    queues: Vec<VecDeque<Queued<T>>>,
    previous: Vec<Option<Capture<T>>>,
    received: Vec<Option<(i64, i64)>>,
    cursor: Vec<Option<usize>>,
    best: Vec<Option<usize>>,
    last_event: Option<i64>,
    emitted: Option<i64>,
    skipped_slot: Option<i64>,
    pending: Option<i64>,
}

impl<T> Selector<T> {
    /// Allocate bounded queues and reusable search scratch for a nonempty input set.
    pub fn new(inputs: usize, config: Config) -> Result<Self, Error> {
        if inputs == 0
            || config.capacity == 0
            || config.wait_us < 0
            || config.skew_us < 0
            || config.max_age_us < config.skew_us
        {
            return Err(Error::Config);
        }
        Ok(Self {
            config,
            queues: (0..inputs)
                .map(|_| VecDeque::with_capacity(config.capacity))
                .collect(),
            previous: (0..inputs).map(|_| None).collect(),
            received: vec![None; inputs],
            cursor: vec![None; inputs],
            best: vec![None; inputs],
            last_event: None,
            emitted: None,
            skipped_slot: None,
            pending: None,
        })
    }

    fn check_clock(&self, now_us: i64) -> Result<(), Error> {
        if now_us < 0 || self.last_event.is_some_and(|last| now_us < last) {
            Err(Error::Clock)
        } else {
            Ok(())
        }
    }

    /// Push one decoded image on the common source clock. Capture times must strictly increase per input, and arrival times must not decrease or precede capture. Arrival determines deadline eligibility, not capture identity.
    pub fn push(
        &mut self,
        camera: usize,
        capture: Capture<T>,
        arrived_us: i64,
    ) -> Result<(), Error> {
        if camera >= self.queues.len()
            || capture.time_us < 0
            || capture.time_us > arrived_us
            || self.received[camera]
                .is_some_and(|(last, arrival)| capture.time_us <= last || arrived_us < arrival)
        {
            return Err(Error::Clock);
        }
        self.received[camera] = Some((capture.time_us, arrived_us));
        let queue = &mut self.queues[camera];
        if queue.len() == self.config.capacity {
            queue.pop_front();
        }
        queue.push_back(Queued {
            capture,
            arrived_us,
        });
        Ok(())
    }

    /// Select a set, or ask the caller to wait. A pending slot cannot move its deadline.
    /// `T::clone` must retain an image handle, not duplicate its pixels.
    pub fn poll(&mut self, slot_us: i64, now_us: i64) -> Result<Option<Selection<T>>, Error>
    where
        T: Clone,
    {
        self.check_clock(now_us)?;
        if slot_us < 0
            || self.emitted.is_some_and(|last| slot_us <= last)
            || self.skipped_slot.is_some_and(|last| slot_us <= last)
            || self.pending.is_some_and(|pending| pending != slot_us)
        {
            return Err(Error::Clock);
        }
        let deadline = slot_us
            .checked_add(self.config.wait_us)
            .ok_or(Error::Clock)?;
        let upper = slot_us
            .checked_add(self.config.skew_us)
            .ok_or(Error::Clock)?;
        self.last_event = Some(now_us);
        self.pending = Some(slot_us);
        let lower = slot_us.saturating_sub(self.config.max_age_us);
        for (queue, previous) in self.queues.iter_mut().zip(&mut self.previous) {
            while queue
                .front()
                .is_some_and(|frame| frame.capture.time_us < lower)
            {
                queue.pop_front();
            }
            // Release unusable owners before waiting: a finite capture pool
            // may need this allocation to produce the replacement image.
            if previous.as_ref().is_some_and(|frame| frame.time_us < lower) {
                *previous = None;
            }
        }
        if now_us < slot_us {
            return Ok(None);
        }
        self.best.fill(None);
        // Sweep ordered queues, advancing only their earliest candidate.
        // Linear sweeps over queued and retained candidates; no Cartesian product of inputs.
        let eligible = |queued: &Queued<T>| {
            queued.capture.time_us <= upper
                && queued.arrived_us <= now_us
                && queued.arrived_us <= deadline
        };
        let mut active = 0;
        for (camera, (cursor, queue)) in self.cursor.iter_mut().zip(&self.queues).enumerate() {
            // A retained image may match better than its replacement. Its age
            // limit still applies; the slot window only filters new captures.
            // Cursor zero denotes that image; queued images start at one.
            *cursor = queue
                .front()
                .filter(|f| eligible(f))
                .map(|_| usize::from(self.previous[camera].is_none()));
            active += usize::from(cursor.is_some());
        }
        let unavailable = |camera| {
            if self.emitted.is_some() {
                Error::Expired(camera)
            } else {
                Error::Missing(camera)
            }
        };
        if active != self.queues.len() {
            if now_us < deadline {
                return Ok(None);
            }
            for (camera, cursor) in self.cursor.iter().enumerate() {
                if cursor.is_none() && self.previous[camera].is_none() {
                    return Err(unavailable(camera));
                }
            }
        }
        let candidate = |camera: usize, cursor: usize| {
            if cursor == 0 {
                self.previous[camera].as_ref()
            } else {
                self.queues[camera]
                    .get(cursor - 1)
                    .filter(|f| eligible(f))
                    .map(|f| &f.capture)
            }
        };
        let mut found = false;
        if active > 0 {
            loop {
                let mut earliest = 0;
                let mut min = i64::MAX;
                let mut max = 0;
                let mut complete = true;
                for camera in 0..self.queues.len() {
                    let Some(cursor) = self.cursor[camera] else {
                        continue;
                    };
                    let Some(frame) = candidate(camera, cursor) else {
                        complete = false;
                        break;
                    };
                    let time = frame.time_us;
                    if time < min {
                        min = time;
                        earliest = camera;
                    }
                    max = max.max(time);
                }
                if !complete {
                    break;
                }
                if max - min <= self.config.skew_us {
                    found = true;
                    self.best.clone_from(&self.cursor);
                }
                *self.cursor[earliest].as_mut().unwrap() += 1;
            }
        }
        if !found && active > 0 {
            if now_us < deadline {
                return Ok(None);
            }
            return Err(Error::Skew);
        }
        // Exhausting the earliest queue need not exclude later compatible peers.
        if let Some(oldest) = self
            .best
            .iter()
            .enumerate()
            .filter_map(|(camera, cursor)| candidate(camera, (*cursor)?).map(|frame| frame.time_us))
            .min()
        {
            for (camera, cursor) in self.best.iter_mut().enumerate() {
                if let Some(cursor) = cursor {
                    while candidate(camera, *cursor + 1)
                        .is_some_and(|frame| frame.time_us - oldest <= self.config.skew_us)
                    {
                        *cursor += 1;
                    }
                }
            }
        }
        if now_us < deadline
            && self
                .best
                .iter()
                .any(|selected| selected.is_none_or(|i| i == 0))
        {
            return Ok(None);
        }
        let at_us = self
            .best
            .iter()
            .enumerate()
            .filter_map(|(camera, selected)| {
                selected
                    .filter(|&i| i > 0)
                    .map(|i| self.queues[camera][i - 1].capture.time_us)
            })
            .fold(slot_us, i64::max);
        // Validate all inputs before consuming any new image.
        for (camera, selected) in self.best.iter().enumerate() {
            let frame = match selected {
                Some(i) if *i > 0 => &self.queues[camera][*i - 1].capture,
                _ => self.previous[camera]
                    .as_ref()
                    .ok_or_else(|| unavailable(camera))?,
            };
            if at_us - frame.time_us > self.config.max_age_us {
                return Err(Error::Expired(camera));
            }
        }
        for (camera, selected) in self.best.iter().enumerate() {
            if let Some(index) = selected.filter(|&i| i > 0) {
                for _ in 1..index {
                    self.queues[camera].pop_front();
                }
                self.previous[camera] = self.queues[camera].pop_front().map(|f| f.capture);
            }
        }
        self.emitted = Some(at_us);
        self.pending = None;
        Ok(Some(Selection {
            at_us,
            captures: self
                .previous
                .iter()
                .map(|capture| capture.as_ref().unwrap().clone())
                .collect(),
        }))
    }

    /// Abandon a pending slot after its deadline, preserving queued captures at or after that slot. Release retained images so resumption requires fresh captures. The caller reports the output gap; this neither changes capture identity nor relaxes age, skew or arrival limits. A skipped slot cannot be polled again, and clock ordering remains intact.
    pub fn skip_pending(&mut self) -> Result<(), Error> {
        let slot = self.pending.ok_or(Error::Clock)?;
        let deadline = slot.checked_add(self.config.wait_us).ok_or(Error::Clock)?;
        if self.last_event.is_none_or(|now| now < deadline) {
            return Err(Error::Clock);
        }
        for previous in &mut self.previous {
            *previous = None;
        }
        for queue in &mut self.queues {
            while queue
                .front()
                .is_some_and(|frame| frame.capture.time_us < slot)
            {
                queue.pop_front();
            }
        }
        self.skipped_slot = Some(slot);
        self.pending = None;
        Ok(())
    }

    /// Whether an input still has unselected images queued, including images for later slots.
    pub fn has_pending(&self, camera: usize) -> bool {
        self.queues
            .get(camera)
            .is_some_and(|queue| !queue.is_empty())
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config => write!(f, "invalid capture synchronization limits"),
            Self::Clock => write!(f, "capture synchronization clock or ordering is invalid"),
            Self::Missing(camera) => write!(f, "camera {camera} has no eligible initial image"),
            Self::Expired(camera) => {
                write!(f, "camera {camera} image exceeded the reuse age limit")
            }
            Self::Skew => write!(f, "new camera images exceed the allowed capture skew"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests;
