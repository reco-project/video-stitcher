//! Linux process/ancestor memory headroom for one-time admission.
//!
//! Limits are hierarchical: <https://docs.kernel.org/admin-guide/cgroup-v2.html>
//! Only ancestors visible in the process's cgroup namespace can be inspected.

use std::io;
use std::path::{Component, Path, PathBuf};

pub(super) fn available_memory(host_available: u64) -> io::Result<u64> {
    headroom(
        host_available,
        &std::fs::read_to_string("/proc/self/cgroup")?,
        &std::fs::read_to_string("/proc/self/mountinfo")?,
        |path| std::fs::read_to_string(path),
    )
}

fn headroom(
    mut available: u64,
    membership: &str,
    mounts: &str,
    read: impl Fn(&Path) -> io::Result<String>,
) -> io::Result<u64> {
    let (mut legacy, mut unified) = (None, None);
    for line in membership.lines() {
        let (_, rest) = line.split_once(':').ok_or_else(invalid)?;
        let (controllers, path) = rest.split_once(':').ok_or_else(invalid)?;
        if controllers.split(',').any(|c| c == "memory") {
            legacy = Some(path);
        } else if controllers.is_empty() {
            unified = Some(path);
        }
    }
    // In hybrid setups, the memory controller can belong to the v1 hierarchy.
    let Some((member, v2)) = legacy.map(|p| (p, false)).or(unified.map(|p| (p, true))) else {
        return Ok(available);
    };
    let member = checked_path(member)?;
    let mut mount = None;
    for line in mounts.lines() {
        let Some((before, after)) = line.split_once(" - ") else {
            continue;
        };
        let mut fs = after.split_whitespace();
        let kind = fs.next();
        let _source = fs.next();
        let options = fs.next().unwrap_or("");
        if !(v2 && kind == Some("cgroup2")
            || !v2 && kind == Some("cgroup") && options.split(',').any(|c| c == "memory"))
        {
            continue;
        }
        let mut fields = before.split_whitespace().skip(3);
        let root = fields.next().ok_or_else(invalid)?;
        let target = fields.next().ok_or_else(invalid)?;
        // A bind-mounted subtree can hide a stricter visible ancestor. Use a
        // full namespace-root mount or require a caller-supplied allowance.
        if root == "/" {
            mount = Some(checked_path(&unescape(target))?);
            break;
        }
    }
    let mount = mount.ok_or_else(|| io::Error::other("memory hierarchy root is not mounted"))?;
    let mut dir = mount.join(member.strip_prefix("/").map_err(|_| invalid())?);
    loop {
        let limit_file = if v2 {
            "memory.max"
        } else {
            "memory.limit_in_bytes"
        };
        match read(&dir.join(limit_file)) {
            Ok(limit) if v2 && limit.trim() == "max" => {}
            Ok(limit) => {
                let limit = number(&limit)?;
                let usage = read(&dir.join(if v2 {
                    "memory.current"
                } else {
                    "memory.usage_in_bytes"
                }))?;
                available = available.min(limit.saturating_sub(number(&usage)?));
            }
            Err(error) if v2 && error.kind() == io::ErrorKind::NotFound => {
                // The real v2 root has no memory.max. Elsewhere it can be
                // absent when memory isn't enabled, but a vanished/inaccessible
                // group must not silently become an unlimited allowance.
                let controllers = read(&dir.join("cgroup.controllers"))?;
                if dir != mount && controllers.split_whitespace().any(|c| c == "memory") {
                    return Err(error);
                }
            }
            Err(error) => return Err(error),
        }
        if dir == mount {
            break;
        }
        if !dir.pop() || !dir.starts_with(&mount) {
            return Err(invalid());
        }
    }
    Ok(available)
}

fn number(value: &str) -> io::Result<u64> {
    value.trim().parse().map_err(|_| invalid())
}

fn checked_path(value: &str) -> io::Result<PathBuf> {
    let path = Path::new(value);
    if !path.is_absolute() || path.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(invalid());
    }
    Ok(path.to_owned())
}

fn unescape(value: &str) -> String {
    value
        .replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
}

fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "invalid cgroup path, membership or memory value",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    const V2: &str = "34 25 0:29 / /cg rw - cgroup2 cgroup2 rw";

    #[test]
    #[ignore = "requires a running systemd user manager with memory delegation"]
    fn live_systemd_memory_limit() {
        const LIMIT: usize = 256 * 1024 * 1024;
        if std::env::var_os("RECO_CGROUP_PROBE_CHILD").is_some() {
            let allowance = super::super::image_budget(None).unwrap();
            println!("256 MiB cgroup: lookahead allowance {allowance} bytes");
            assert!(allowance > 0 && allowance <= LIMIT / 2);
            return;
        }
        let status = std::process::Command::new("systemd-run")
            .args([
                "--user",
                "--pipe",
                "--wait",
                "--collect",
                "--quiet",
                "--property=MemoryMax=268435456",
                "--setenv=RECO_CGROUP_PROBE_CHILD=1",
            ])
            .arg(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "lookahead::cgroup::tests::live_systemd_memory_limit",
                "--ignored",
                "--nocapture",
            ])
            .status()
            .unwrap();
        assert!(status.success(), "cgroup child probe failed: {status}");
    }

    fn probe(member: &str, mounts: &str, files: &[(&str, &str)]) -> io::Result<u64> {
        let files: HashMap<_, _> = files.iter().copied().collect();
        headroom(10_000, member, mounts, |path| {
            files
                .get(path.to_str().unwrap())
                .map(|s| s.to_string())
                .ok_or_else(|| io::Error::from(io::ErrorKind::NotFound))
        })
    }

    #[test]
    fn nested_v2_accounts_ancestor_usage_not_only_leaf_limit() {
        let files = [
            ("/cg/job/child/memory.max", "max"),
            ("/cg/job/memory.max", "1000"),
            ("/cg/job/memory.current", "800"),
            ("/cg/cgroup.controllers", "memory cpu"),
        ];
        assert_eq!(probe("0::/job/child", V2, &files).unwrap(), 200);
        let mut files = files.to_vec();
        files[0].1 = "100";
        files.push(("/cg/job/child/memory.current", "150"));
        assert_eq!(probe("0::/job/child", V2, &files).unwrap(), 0);
    }

    #[test]
    fn namespaced_root_and_host_limit_are_respected() {
        assert_eq!(
            probe(
                "0::/",
                V2,
                &[("/cg/memory.max", "500"), ("/cg/memory.current", "100"),]
            )
            .unwrap(),
            400
        );
        assert_eq!(
            probe(
                "0::/",
                V2,
                &[("/cg/memory.max", "50000"), ("/cg/memory.current", "100"),]
            )
            .unwrap(),
            10_000
        );
        assert_eq!(
            probe("0::/", V2, &[("/cg/cgroup.controllers", "cpu")]).unwrap(),
            10_000
        );
        assert_eq!(probe("2:cpu:/job", "", &[]).unwrap(), 10_000);
    }

    #[test]
    fn hybrid_prefers_v1_memory_and_walks_ancestors() {
        let mounts = format!("{V2}\n35 25 0:30 / /mem rw - cgroup cgroup rw,memory");
        assert_eq!(
            probe(
                "0::/wrong\n4:memory:/job",
                &mounts,
                &[
                    ("/mem/job/memory.limit_in_bytes", "10000"),
                    ("/mem/job/memory.usage_in_bytes", "100"),
                    ("/mem/memory.limit_in_bytes", "1000"),
                    ("/mem/memory.usage_in_bytes", "900"),
                ]
            )
            .unwrap(),
            100
        );
    }

    #[test]
    fn mount_escapes_are_decoded_once() {
        let mounts = "34 25 0:29 / /cg\\040space\\134040 rw - cgroup2 cgroup2 rw";
        assert_eq!(
            probe(
                "0::/",
                mounts,
                &[
                    ("/cg space\\040/memory.max", "500"),
                    ("/cg space\\040/memory.current", "100"),
                ]
            )
            .unwrap(),
            400
        );
    }

    #[test]
    fn unresolved_or_malformed_limits_never_use_host_allowance() {
        for (member, mounts, files) in [
            ("0::/gone", V2, vec![]),
            ("0::/../outside", V2, vec![]),
            ("broken", V2, vec![]),
            (
                "0::/",
                "34 25 0:29 /hidden /cg rw - cgroup2 cgroup2 rw",
                vec![],
            ),
            ("0::/", V2, vec![("/cg/memory.max", "broken")]),
            ("0::/", V2, vec![("/cg/memory.max", "100")]),
            (
                "0::/job",
                V2,
                vec![("/cg/job/cgroup.controllers", "memory")],
            ),
        ] {
            assert!(probe(member, mounts, &files).is_err(), "{member}");
        }
        assert!(
            headroom(10_000, "0::/", V2, |_| Err(
                io::ErrorKind::PermissionDenied.into()
            ))
            .is_err()
        );
    }
}
