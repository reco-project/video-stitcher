//! Local package asset server and authenticated live-editor bridge.

use std::io::{Read, Write};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream, UdpSocket};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use percent_encoding::percent_decode_str;
use thiserror::Error;

const SDK: &[u8] = include_bytes!("../../../scoreboards/sdk/reco-scoreboard.js");
const EDITOR_STATE_PATH: &str = "/__reco/editor-state";
const MAX_REQUEST_BYTES: usize = 64 * 1024;

#[derive(Default)]
struct EditorStateInner {
    version: u64,
    json: Option<String>,
}

#[derive(Clone, Default)]
pub(crate) struct EditorState {
    inner: Arc<Mutex<EditorStateInner>>,
}

impl EditorState {
    pub(crate) fn newer_than(&self, version: u64) -> Option<(u64, String)> {
        let state = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        (state.version > version)
            .then(|| (state.version, state.json.clone()))
            .and_then(|(version, json)| json.map(|json| (version, json)))
    }

    fn current_json(&self) -> String {
        self.current().unwrap_or_else(|| "null".into())
    }

    pub(crate) fn current(&self) -> Option<String> {
        self.inner
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .json
            .clone()
    }

    pub(crate) fn replace(&self, json: String) {
        let mut state = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        state.version = state.version.wrapping_add(1).max(1);
        state.json = Some(json);
    }
}

pub(crate) struct LocalPackageServer {
    address: SocketAddr,
    network_address: Option<SocketAddr>,
    editor_token: String,
    editor_state: EditorState,
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
        let listener = TcpListener::bind((Ipv4Addr::UNSPECIFIED, 0)).map_err(ServerError::Bind)?;
        listener.set_nonblocking(true).map_err(ServerError::Bind)?;
        let port = listener.local_addr().map_err(ServerError::Bind)?.port();
        let address = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
        let network_address = local_network_ipv4().map(|address| SocketAddr::from((address, port)));
        let mut token_bytes = [0_u8; 24];
        getrandom::fill(&mut token_bytes)
            .map_err(|error| ServerError::Random(error.to_string()))?;
        let editor_token = token_bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let editor_state = EditorState::default();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let worker_token = editor_token.clone();
        let worker_editor_state = editor_state.clone();
        let thread = std::thread::Builder::new()
            .name("scoreboard-assets".into())
            .spawn(move || {
                run(
                    listener,
                    canonical_root,
                    worker_token,
                    worker_editor_state,
                    worker_stop,
                )
            })
            .map_err(ServerError::Spawn)?;
        Ok(Self {
            address,
            network_address,
            editor_token,
            editor_state,
            stop,
            thread: Some(thread),
        })
    }

    pub(crate) fn url_for(&self, entry: &str) -> String {
        format!("http://{}/{}", self.address, entry.replace('\\', "/"))
    }

    pub(crate) fn editor_url_for(&self, entry: &str) -> String {
        self.editor_url_for_address(self.address, entry)
    }

    pub(crate) fn network_editor_url_for(&self, entry: &str) -> Option<String> {
        self.network_address
            .map(|address| self.editor_url_for_address(address, entry))
    }

    fn editor_url_for_address(&self, address: SocketAddr, entry: &str) -> String {
        let separator = if entry.contains('?') { '&' } else { '?' };
        format!(
            "http://{address}/{}{separator}recoEditorToken={}",
            entry.replace('\\', "/"),
            self.editor_token
        )
    }

    pub(crate) fn editor_state(&self) -> EditorState {
        self.editor_state.clone()
    }
}

fn local_network_ipv4() -> Option<Ipv4Addr> {
    // A connected UDP socket reveals the interface selected by the OS without
    // sending traffic. The documentation-only destination never needs to be
    // reachable for route selection to succeed.
    let socket = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0)).ok()?;
    socket.connect((Ipv4Addr::new(192, 0, 2, 1), 9)).ok()?;
    match socket.local_addr().ok()?.ip() {
        IpAddr::V4(address) if !address.is_loopback() && !address.is_unspecified() => Some(address),
        _ => None,
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

fn run(
    listener: TcpListener,
    package_root: PathBuf,
    editor_token: String,
    editor_state: EditorState,
    stop: Arc<AtomicBool>,
) {
    while !stop.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _)) => serve(stream, &package_root, &editor_token, &editor_state),
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

fn serve(
    mut stream: TcpStream,
    package_root: &Path,
    editor_token: &str,
    editor_state: &EditorState,
) {
    // Accepted sockets inherit O_NONBLOCK from the listener on macOS. Chrome
    // may split headers across packets, so switch the connection back to a
    // bounded blocking read before parsing the complete request.
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let request = match read_request(&mut stream) {
        Ok(request) => request,
        Err(error) => {
            respond(&mut stream, 400, "text/plain", error.as_bytes(), "GET");
            return;
        }
    };
    let Some(header_end) = find_header_end(&request) else {
        return;
    };
    let headers = String::from_utf8_lossy(&request[..header_end]);
    let mut lines = headers.lines();
    let Some(line) = lines.next() else { return };
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default();
    let target = parts.next().unwrap_or_default();
    let editor_authorized = target
        .split_once('?')
        .map(|(_, query)| {
            query
                .split('&')
                .filter_map(|pair| pair.split_once('='))
                .any(|(key, value)| key == "recoEditorToken" && value == editor_token)
        })
        .unwrap_or(false)
        || lines.clone().any(|line| {
            line.split_once(':').is_some_and(|(name, value)| {
                name.eq_ignore_ascii_case("x-reco-editor-token") && value.trim() == editor_token
            })
        });

    let raw_path = target.split('?').next().unwrap_or_default();
    if raw_path == EDITOR_STATE_PATH {
        if !editor_authorized {
            respond(&mut stream, 403, "text/plain", b"Forbidden", method);
            return;
        }
        match method {
            "GET" | "HEAD" => {
                let json = editor_state.current_json();
                respond(
                    &mut stream,
                    200,
                    "application/json; charset=utf-8",
                    json.as_bytes(),
                    method,
                );
            }
            "PUT" => {
                let body = &request[header_end + 4..];
                match serde_json::from_slice::<serde_json::Value>(body) {
                    Ok(value) => {
                        editor_state.replace(value.to_string());
                        respond(&mut stream, 204, "text/plain", b"", method);
                    }
                    Err(_) => respond(&mut stream, 400, "text/plain", b"Invalid JSON", method),
                }
            }
            _ => respond(
                &mut stream,
                405,
                "text/plain",
                b"Method not allowed",
                method,
            ),
        }
        return;
    }
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

fn read_request(stream: &mut TcpStream) -> Result<Vec<u8>, String> {
    let mut request = Vec::with_capacity(8192);
    let mut chunk = [0_u8; 4096];
    loop {
        let length = stream
            .read(&mut chunk)
            .map_err(|error| format!("Bad request: {error}"))?;
        if length == 0 {
            break;
        }
        request.extend_from_slice(&chunk[..length]);
        if request.len() > MAX_REQUEST_BYTES {
            return Err("Bad request: request is too large".into());
        }
        if let Some(header_end) = find_header_end(&request) {
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let content_length = headers
                .lines()
                .filter_map(|line| line.split_once(':'))
                .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                .and_then(|(_, value)| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            if request.len() >= header_end + 4 + content_length {
                request.truncate(header_end + 4 + content_length);
                return Ok(request);
            }
        }
    }
    Ok(request)
}

fn find_header_end(request: &[u8]) -> Option<usize> {
    request.windows(4).position(|window| window == b"\r\n\r\n")
}

fn respond(stream: &mut TcpStream, status: u16, content_type: &str, body: &[u8], method: &str) {
    let reason = match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        403 => "Forbidden",
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
    #[error("cannot create scoreboard editor token: {0}")]
    Random(String),
    #[error("cannot start scoreboard asset server: {0}")]
    Spawn(std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(server: &LocalPackageServer, target: &str) -> String {
        request_at(server.address, target)
    }

    fn request_at(address: SocketAddr, target: &str) -> String {
        let mut stream = TcpStream::connect(address).unwrap();
        write!(stream, "GET {target} HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        response
    }

    fn put_editor_state(server: &LocalPackageServer, token: &str, body: &str) -> String {
        let mut stream = TcpStream::connect(server.address).unwrap();
        write!(
            stream,
            "PUT {EDITOR_STATE_PATH} HTTP/1.1\r\nHost: localhost\r\nX-Reco-Editor-Token: {token}\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
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

    #[test]
    fn editor_state_requires_token_and_publishes_versioned_json() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("index.html"), "scoreboard").unwrap();
        let server = LocalPackageServer::start(temp.path().to_path_buf()).unwrap();
        let state = server.editor_state();

        let forbidden = put_editor_state(&server, "wrong", r#"{"score":1}"#);
        assert!(forbidden.starts_with("HTTP/1.1 403 Forbidden"));
        assert!(state.newer_than(0).is_none());

        let accepted = put_editor_state(&server, &server.editor_token, r#"{"score":2}"#);
        assert!(accepted.starts_with("HTTP/1.1 204 No Content"));
        let (version, json) = state.newer_than(0).unwrap();
        assert_eq!(version, 1);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&json).unwrap()["score"],
            2
        );
    }

    #[test]
    fn editor_is_reachable_at_authenticated_local_network_url() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("index.html"), "scoreboard").unwrap();
        let server = LocalPackageServer::start(temp.path().to_path_buf()).unwrap();
        let Some(network_address) = server.network_address else {
            eprintln!("No local network address available; skipping LAN reachability check");
            return;
        };

        let editor_url = server.network_editor_url_for("index.html?debug=1").unwrap();
        assert!(editor_url.contains(&network_address.to_string()));
        assert!(editor_url.contains(&server.editor_token));
        assert!(request_at(network_address, "/index.html").starts_with("HTTP/1.1 200 OK"));
    }
}
