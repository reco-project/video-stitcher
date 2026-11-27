//! End-to-end bounds on the source frames used for each camera decision.

use super::*;
use crate::detect::pipeline_event::{PipelineEvent, PipelineEventSink};

struct CountedSource {
    source: MockSource,
    read: Arc<AtomicU64>,
}

impl FrameSource for CountedSource {
    fn info(&self) -> SourceInfo {
        self.source.info()
    }

    fn next_frame(&mut self) -> Result<Option<crate::source::SourceFrame>, SourceError> {
        let frame = self.source.next_frame()?;
        if frame.is_some() {
            self.read.fetch_add(1, Ordering::Relaxed);
        }
        Ok(frame)
    }
}

struct Decisions {
    read: Arc<AtomicU64>,
    poses: Arc<Mutex<Vec<(u64, WorldPose)>>>,
}

impl PipelineEventSink for Decisions {
    fn emit(&mut self, event: PipelineEvent) {
        if let PipelineEvent::PanDecision { pose, .. } = event {
            self.poses
                .lock()
                .unwrap()
                .push((self.read.load(Ordering::Relaxed), pose));
        }
    }
}

struct WindowPanner;
impl Panner for WindowPanner {
    fn decide(&mut self, world: &WorldState, ctx: &PanContext<'_>) -> WorldPose {
        self.decide_with_lookahead(world, &[], ctx)
    }

    fn decide_with_lookahead(
        &mut self,
        _: &WorldState,
        future: &[WorldState],
        ctx: &PanContext<'_>,
    ) -> WorldPose {
        WorldPose {
            yaw: ctx.frame_index as f32 * 0.01,
            fov_degrees: 40.0 + future.len() as f32,
            ..Default::default()
        }
    }
}

fn replay(length: u64, depth: usize) -> Vec<WorldPose> {
    let read = Arc::new(AtomicU64::new(0));
    let poses = Arc::new(Mutex::new(Vec::new()));
    let mut session = build_cpu_session();
    session.core_mut().set_constrained_look(false);
    session.set_lookahead(depth, usize::MAX);
    session.set_panner(Box::new(WindowPanner));
    session.set_event_sink(Box::new(Decisions {
        read: read.clone(),
        poses: poses.clone(),
    }));
    let mut source = CountedSource {
        source: MockSource::new(length),
        read,
    };
    session
        .run(&mut source, u64::MAX, &AtomicBool::new(false), None)
        .unwrap();
    let observations = poses.lock().unwrap();
    assert_eq!(observations.len(), length as usize);
    observations
        .iter()
        .enumerate()
        .map(|(index, &(read, pose))| {
            let expected = (index as u64 + depth as u64 + 1).min(length);
            assert_eq!(
                read, expected,
                "read horizon at frame {index} with depth {depth}"
            );
            assert_eq!(
                pose.fov_degrees,
                40.0 + (expected - index as u64 - 1) as f32
            );
            pose
        })
        .collect()
}

#[test]
fn source_reads_and_decisions_respect_one_lookahead_window() {
    for depth in [0, 1, 4, 45] {
        let full = replay(52, depth);
        let prefix = replay(3 + depth as u64, depth);
        assert_eq!(full[..3], prefix[..3]);
        replay(1, depth);
    }
}
