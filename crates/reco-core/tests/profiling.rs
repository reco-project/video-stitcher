//! The exported profiling macro must not resolve dependencies in its caller.

// A caller may have its own tracing module or no tracing dependency at all.
mod tracing {}

#[test]
fn profiling_scope_is_independent_of_caller_imports() {
    reco_core::profile_scope!("external_scope");
}

#[cfg(not(feature = "profiling"))]
#[test]
fn disabled_profiling_does_not_evaluate_its_argument() {
    reco_core::profile_scope!(panic!("disabled profiling evaluated the span name"));
}

#[cfg(feature = "profiling")]
mod enabled {
    use ::tracing::{Event, Metadata, Subscriber, span};
    use std::sync::{Arc, Mutex};

    struct Recorder(Arc<Mutex<Vec<&'static str>>>);

    impl Subscriber for Recorder {
        fn enabled(&self, _: &Metadata<'_>) -> bool {
            true
        }

        fn new_span(&self, attributes: &span::Attributes<'_>) -> span::Id {
            self.0.lock().unwrap().push(attributes.metadata().name());
            span::Id::from_u64(1)
        }

        fn record(&self, _: &span::Id, _: &span::Record<'_>) {}
        fn record_follows_from(&self, _: &span::Id, _: &span::Id) {}
        fn event(&self, _: &Event<'_>) {}

        fn enter(&self, _: &span::Id) {
            self.0.lock().unwrap().push("enter");
        }

        fn exit(&self, _: &span::Id) {
            self.0.lock().unwrap().push("exit");
        }
    }

    #[test]
    fn span_covers_the_callers_scope() {
        let events = Arc::new(Mutex::new(Vec::new()));
        ::tracing::subscriber::with_default(Recorder(events.clone()), || {
            {
                reco_core::profile_scope!("measured_scope");
                assert_eq!(*events.lock().unwrap(), ["measured_scope", "enter"]);
                events.lock().unwrap().push("work");
            }
            assert_eq!(
                *events.lock().unwrap(),
                ["measured_scope", "enter", "work", "exit"]
            );
        });
    }
}
