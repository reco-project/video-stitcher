// Shared forward KB4 optical map. Plane geometry and support are caller-owned.
fn lens_uv(extended: vec2<f32>, intrinsics: vec4<f32>, distortion: vec4<f32>, correction: f32) -> vec2<f32> {
    let normalized = (extended - intrinsics.zw) / intrinsics.xy;
    let radius = length(normalized);
    let theta = atan(radius);
    let t2 = theta * theta;
    let d = distortion * correction;
    let distorted = theta * (1.0 + t2 * (d.x + t2 * (d.y + t2 * (d.z + t2 * d.w))));
    var scale = 1.0;
    if radius > 0.0 { scale = distorted / radius; }
    return intrinsics.xy * normalized * scale + intrinsics.zw;
}
