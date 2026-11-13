using System.Net.Http.Json;
namespace Posegraph.Backend;

public sealed partial class BackendClient
{
    public sealed record LiveCamera(int Index, string Name, string? Path);
    public async Task<LiveCamera[]> LiveCamerasAsync()
    {
        using var response = await _http.GetAsync(Url("/live/cameras"));
        if (response.StatusCode == System.Net.HttpStatusCode.NotFound)
            throw new InvalidOperationException("当前后端不支持 Live，请更新实际使用的后端并重启（仅更新插件 DLL 不够）。");
        return (await ReadAsync<LiveCamera[]>(response))!;
    }
    public sealed record LiveSession(string Id, bool FaceSupported);
    public sealed record LiveFrame(long Sequence, bool Tracked, string? Pose, string? BodyPose, bool FaceSupported, string? Preview, double FrameMs, float[][]? Lines, float[]? FaceBox, string[]? HandsTracked, bool TorsoTracked, bool FullPose, float[]? Framing, string? CalibrationMessage);
    public async Task<LiveSession> StartLiveAsync(string reference, string race, int camera, float smoothing, float faceStrength, Dictionary<string, float> faceGains, float mouthLimit, string? cameraName, string? cameraPath)
    {
        using var response = await _http.PostAsJsonAsync(Url("/live/start"), new { reference, race, camera, smoothing, face_strength = faceStrength, face_gains = faceGains, mouth_limit = mouthLimit, camera_name = cameraName, camera_path = cameraPath });
        return (await ReadAsync<LiveSession>(response))!;
    }
    public async Task<LiveFrame> LiveFrameAsync(string id, CancellationToken ct, float faceStrength, Dictionary<string, float> faceGains, float mouthLimit)
    {
        using var response = await _http.PostAsJsonAsync(Url($"/live/{id}/frame"), new { face_strength = faceStrength, face_gains = faceGains, mouth_limit = mouthLimit }, ct);
        return (await ReadAsync<LiveFrame>(response))!;
    }
    public Task StopLiveAsync(string id) => PostAsync($"/live/{id}/stop", null);
    public Task CalibrateLiveAsync(string id) => PostAsync($"/live/{id}/calibrate", null);
}
