//! Loopback-only, read-only package asset server.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use percent_encoding::percent_decode_str;
use thiserror::Error;

const SDK: &[u8] = include_bytes!("../../../scoreboards/sdk/reco-scoreboard.js");

pub(crate) struct LocalPackageServer {
    address: SocketAddr,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl LocalPackageServer {
    pub(crate) fn start(package_root: PathBuf) -> Result<Self, ServerError> {
        let canonical_root =
            package_root
                .canonicalize()
                .map_err(|source| ServerError::PackageRoot {
                    path: package_root,
                    source,
                })?;
        let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(ServerError::Bind)?;
        listener.set_nonblocking(true).map_err(ServerError::Bind)?;
        let address = listener.local_addr().map_err(ServerError::Bind)?;
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let thread = std::thread::Builder::new()
            .name("scoreboard-assets".into())
            .spawn(move || run(listener, canonical_root, worker_stop))
            .map_err(ServerError::Spawn)?;
        Ok(Self {
            address,
            stop,
            thread: Some(thread),
        })
    }

    pub(crate) fn url_for(&self, entry: &str) -> String {
        format!("http://{}/{}", self.address, entry.replace('\\', "/"))
    }
}

impl Drop for LocalPackageServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn run(listener: TcpListener, package_root: PathBuf, stop: Arc<AtomicBool>) {
    while !stop.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _)) => serve(stream, &package_root),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(2));
            }
            Err(error) => {
                log::error!("scoreboard asset server stopped: {error}");
                return;
            }
        }
    }
}

fn serve(mut stream: TcpStream, package_root: &Path) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let mut request = [0_u8; 8192];
    let Ok(length) = stream.read(&mut request) else {
        return;
    };
    let request = String::from_utf8_lossy(&request[..length]);
    let Some(line) = request.lines().next() else {
        return;
    };
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default();
    let target = parts.next().unwrap_or_default();
    if !matches!(method, "GET" | "HEAD") {
        respond(
            &mut stream,
            405,
            "text/plain",
            b"Method not allowed",
            method,
        );
        return;
    }
    let raw_path = target.split('?').next().unwrap_or_default();
    let Ok(decoded) = percent_decode_str(raw_path).decode_utf8() else {
        respond(&mut stream, 400, "text/plain", b"Bad request", method);
        return;
    };
    let relative = decoded.trim_start_matches('/');
    if relative == "sdk/reco-scoreboard.js" {
        respond(
            &mut stream,
            200,
            "text/javascript; charset=utf-8",
            SDK,
            method,
        );
        return;
    }
    let relative_path = Path::new(relative);
    if relative_path.as_os_str().is_empty()
        || relative_path.is_absolute()
        || relative_path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        log::warn!("scoreboard asset request rejected: {relative}");
        respond(&mut stream, 404, "text/plain", b"Not found", method);
        return;
    }
    let candidate = package_root.join(relative_path);
    let Ok(canonical) = candidate.canonicalize() else {
        log::warn!("scoreboard asset is missing: {relative}");
        respond(&mut stream, 404, "text/plain", b"Not found", method);
        return;
    };
    if !canonical.starts_with(package_root) || !canonical.is_file() {
        log::warn!("scoreboard asset request escaped its package: {relative}");
        respond(&mut stream, 404, "text/plain", b"Not found", method);
        return;
    }
    match std::fs::read(&canonical) {
        Ok(body) => respond(&mut stream, 200, content_type(&canonical), &body, method),
        Err(error) => {
            log::warn!("cannot read scoreboard asset {relative}: {error}");
            respond(&mut stream, 404, "text/plain", b"Not found", method);
        }
    }
}

fn respond(stream: &mut TcpStream, status: u16, content_type: &str, body: &[u8], method: &str) {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        _ => "Error",
    };
    let header = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(header.as_bytes());
    if method != "HEAD" {
        let _ = stream.write_all(body);
    }
}

fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("json") => "application/json; charset=utf-8",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("svg") => "image/svg+xml",
        Some("woff") => "font/woff",
        Some("woff2") => "font/woff2",
        _ => "application/octet-stream",
    }
}

#[derive(Debug, Error)]
pub(crate) enum ServerError {
    #[error("cannot resolve package root {path}: {source}")]
    PackageRoot {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("cannot bind scoreboard asset server: {0}")]
    Bind(std::io::Error),
    #[error("cannot start scoreboard asset server: {0}")]
    Spawn(std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(server: &LocalPackageServer, target: &str) -> String {
        let mut stream = TcpStream::connect(server.address).unwrap();
        write!(stream, "GET {target} HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        response
    }

    #[test]
    fn content_types_cover_scoreboard_assets() {
        assert_eq!(
            content_type(Path::new("index.html")),
            "text/html; charset=utf-8"
        );
        assert_eq!(content_type(Path::new("logo.png")), "image/png");
        assert_eq!(content_type(Path::new("font.woff2")), "font/woff2");
    }

    #[test]
    fn serves_contained_assets_and_rejects_encoded_traversal() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("index.html"), "scoreboard").unwrap();
        let server = LocalPackageServer::start(temp.path().to_path_buf()).unwrap();

        let valid = request(&server, "/index.html");
        assert!(valid.starts_with("HTTP/1.1 200 OK"));
        assert!(valid.ends_with("scoreboard"));

        let traversal = request(&server, "/%2e%2e/secret.txt");
        assert!(traversal.starts_with("HTTP/1.1 404 Not Found"));
    }
}
