using System.Globalization;
using System.Numerics;
using System.Text.Json;
using Dalamud.Game.ClientState.Objects.Types;

namespace Posegraph;

/// <summary>Camera composition estimated from shoulder span; all calls on framework thread.</summary>
public sealed class LiveScene(BrioBridge brio)
{
    private (Vector3 Position, Quaternion Rotation, float Scale)? _original;
    private GposeCamera.SavedSettings? _camera;
    private float? _initialDistance;
    private int _samples, _settle;
    private Vector3 _sumCenter, _sumOffset;
    private float _sumDistance;
    private Vector3 _lockedCamera;
    private Vector3 _offset;
    private float _fov;
    public bool Active => _original is not null;

    public void Begin(IGameObject actor)
    {
        var (p, q, scale) = brio.GetModelTransform(actor);
        if (p is null || q is null || GposeCamera.Projection() is not { } projection)
            throw new InvalidOperationException("无法读取人物位置或 GPose 镜头。");
        _original = (p.Value, q.Value, scale?.X ?? 1f);
        _camera = GposeCamera.SaveSettings();
        _fov = projection.FovY;
        Recalibrate();
    }

    private static Vector3? Bone(JsonElement bones, string name)
    {
        if (!bones.TryGetProperty(name, out var b)) return null;
        var parts = b.GetProperty("Position").GetString()!.Split(',');
        if (parts.Length != 3) return null;
        return new Vector3(float.Parse(parts[0], CultureInfo.InvariantCulture), float.Parse(parts[1], CultureInfo.InvariantCulture), float.Parse(parts[2], CultureInfo.InvariantCulture));
    }

    public void Apply(IGameObject actor, string pose, float[]? framing)
    {
        if (_original is not { } original || framing is not { Length: 4 } f || f.Any(x => !float.IsFinite(x)) || f[2] < .04f) return;
        using var doc = JsonDocument.Parse(pose);
        var bones = doc.RootElement.GetProperty("Bones");
        if (Bone(bones, "j_ude_a_l") is not { } left || Bone(bones, "j_ude_a_r") is not { } right) return;
        var center = (left + right) * .5f * original.Scale;
        var span = Vector3.Distance(left, right) * original.Scale;
        if (span < .02f || GposeCamera.Projection() is not { } projection) return;
        // Collect a short stable window, then freeze placement. Shoulder width
        // changes with turning and detection noise; it is not a depth sensor.
        if (_initialDistance is null)
        {
            var spanY = f[2] * Math.Clamp(f[3], .5f, 3f);
            var distance = Math.Clamp(span / (2 * MathF.Tan(_fov / 2) * spanY), .35f, 8f);
            if (_samples > 0 && MathF.Abs(distance - _sumDistance / _samples) > _sumDistance / _samples * .25f)
            { Recalibrate(); return; }
            var halfHeight = distance * MathF.Tan(_fov / 2);
            _sumCenter += center;
            _sumDistance += distance;
            _sumOffset += new Vector3((f[0] - .5f) * 2 * halfHeight * projection.Aspect, (.5f - f[1]) * 2 * halfHeight, 0);
            if (++_samples < 15) return;
            _initialDistance = _sumDistance / _samples;
            _offset = Vector3.Clamp(_sumOffset / _samples, new Vector3(-2, -1, 0), new Vector3(2, 1, 0));
            var position = original.Position + Vector3.Transform(_offset, original.Rotation);
            if (!brio.SetModelTransform(actor, position, original.Rotation)) throw new InvalidOperationException("Live 无法移动角色。");
            _lockedCamera = original.Position + Vector3.Transform(_sumCenter / _samples + new Vector3(0, 0, _initialDistance.Value), original.Rotation);
            GposeCamera.PlacePosition(_lockedCamera, _fov);
            _settle = 12;
        }
        if (_settle > 0)
        {
            GposeCamera.SteerView(Vector3.Transform(-Vector3.UnitZ, original.Rotation), Vector3.Transform(Vector3.UnitY, original.Rotation));
            _settle--;
        }
    }

    public void Recalibrate()
    {
        _initialDistance = null; _samples = _settle = 0;
        _sumCenter = _sumOffset = _offset = Vector3.Zero;
        _sumDistance = 0;
    }

    public void Restore(IGameObject? actor, bool restoreCamera)
    {
        if (_original is { } original && actor is not null) brio.SetModelTransform(actor, original.Position, original.Rotation);
        if (restoreCamera && _camera is { } camera) GposeCamera.RestoreSettings(camera);
        _original = null; _camera = null; Recalibrate();
    }
}
