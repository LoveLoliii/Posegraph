using System.Numerics;
using System.Text.Json;
using System.Text.Json.Nodes;
using Dalamud.Bindings.ImGui;
using Dalamud.Interface.Windowing;
using Dalamud.Interface.Textures.TextureWraps;
using Dalamud.Plugin.Services;
using Dalamud.Plugin;
using Posegraph.Backend;

namespace Posegraph.UI;

public sealed class LiveWindow : Window, IDisposable
{
    private readonly MainWindow _main;
    private readonly ITextureProvider _textures;
    private IDalamudTextureWrap? _preview;
    private bool _mirrorPreview = true;
    private bool _showTracking = true;
    private float[][]? _trackingLines;
    private float[]? _faceBox;
    private string _bodyStatus = "";
    private string? _previewError;
    private readonly IFramework _framework;
    private readonly PoseApplier _applier;
    private readonly BrioBridge _brio;
    private readonly BrioDirectPose _direct = new();
    private readonly LiveScene _scene;
    private bool _matchScene = true;
    private CancellationTokenSource? _cancel;
    private bool _calibrating;
    private long _calibrationStarted;
    private Task? _run;
    private string? _session;
    private string? _backup;
    private PoseApplier.ActorIdentity? _actor;
    private int _camera;
    private BackendClient.LiveCamera[] _cameras = [];
    private bool _loadingCameras;
    public override void OnOpen() => _ = RefreshCamerasAsync();
    private async Task RefreshCamerasAsync()
    {
        if (_loadingCameras || _busy) return;
        _loadingCameras = true;
        try
        {
            var previous = _cameras.FirstOrDefault(c => c.Index == _camera)?.Path;
            _cameras = await _main.Client.LiveCamerasAsync();
            _camera = _cameras.FirstOrDefault(c => previous is not null && c.Path == previous)?.Index
                ?? _cameras.FirstOrDefault()?.Index ?? 0;
            _status = _cameras.Length == 0 ? "未发现摄像头，请连接设备后刷新。" : "请选择摄像头，然后开始捕捉。";
        }
        catch (Exception ex) { _cameras = []; _status = ex.Message; }
        finally { _loadingCameras = false; }
    }
    private float _smoothing = 0.35f;
    private readonly Configuration _config;
    private readonly IDalamudPluginInterface _pi;
    private string _status = "准备就绪：正对摄像头，开始后第一帧作为头部中性姿态。";
    private bool _busy;
    private bool _disposed;
    private bool _paused;
    private bool _faceSupported;
    private bool _poseChanged;
    private double _fps, _backendMs;
    private long _lastFrameTick;
    public Func<bool>? VideoPlaying { get; set; }

    public LiveWindow(MainWindow main, IFramework framework, PoseApplier applier, BrioBridge brio, ITextureProvider textures, Configuration config, IDalamudPluginInterface pi)
        : base("Posegraph Live###PosegraphLive", ImGuiWindowFlags.AlwaysVerticalScrollbar)
    {
        _main = main; _framework = framework; _applier = applier; _brio = brio;
        _textures = textures;
        _scene = new LiveScene(brio);
        _config = config; _pi = pi;
        _config.LiveFaceStrength ??= _config.VideoFaceStrength;
        _config.LiveFaceGains ??= new(_config.VideoFaceGains);
        _pi.SavePluginConfig(_config);
        Size = new Vector2(520, 620);
        SizeCondition = ImGuiCond.FirstUseEver;
    }

    public override void Draw()
    {
        ImGui.TextUnformatted("Live · 头部、躯干、手指与表情捕捉（实验）");
        ImGui.Separator();
        DrawPreview();
        ImGui.BeginDisabled(_busy);
        if (ImGui.BeginCombo("摄像头", _cameras.FirstOrDefault(c => c.Index == _camera)?.Name ?? "未发现设备"))
        {
            foreach (var camera in _cameras)
            {
                if (ImGui.Selectable($"{camera.Name}##camera{camera.Index}", _camera == camera.Index)) _camera = camera.Index;
                if (ImGui.IsItemHovered() && camera.Path is not null) ImGui.SetTooltip(camera.Path);
            }
            ImGui.EndCombo();
        }
        ImGui.SameLine();
        ImGui.BeginDisabled(_loadingCameras);
        if (ImGui.Button(_loadingCameras ? "正在查找…" : "刷新设备")) _ = RefreshCamerasAsync();
        ImGui.EndDisabled();
        ImGui.Checkbox("校准时匹配位置与镜头（之后固定）", ref _matchScene);
        ImGui.SliderFloat("响应速度", ref _smoothing, 0.05f, 1f);
        ImGui.BeginDisabled(_loadingCameras || _cameras.Length == 0);
        if (ImGui.Button("开始捕捉")) _run = RunAsync();
        ImGui.EndDisabled();
        ImGui.EndDisabled();
        ImGui.SameLine();
        ImGui.BeginDisabled(!_busy);
        if (ImGui.Button("停止并恢复")) _cancel?.Cancel();
        ImGui.EndDisabled();
        if (_busy)
        {
            ImGui.Checkbox("暂停跟随", ref _paused);
            ImGui.SameLine();
            ImGui.BeginDisabled(_calibrating || _session is null);
            if (ImGui.Button("校准动作（2秒后）")) _ = CalibrateAsync();
            ImGui.EndDisabled();
            if (_calibrating)
            {
                var remaining = Math.Max(0, 2 - System.Diagnostics.Stopwatch.GetElapsedTime(_calibrationStarted).TotalSeconds);
                ImGui.TextUnformatted(remaining > 0 ? $"{remaining:F1} 秒后校准：双手向前举，手指向上、掌心朝摄像头…" : "正在校准…");
            }
        }
        DrawFaceWeights();
        ImGui.TextWrapped(_status);
        if (_busy) ImGui.TextWrapped(_bodyStatus);
        if (_busy && _fps > 0)
            ImGui.TextDisabled($"接收帧率 {_fps:F1} FPS · 后端处理 {_backendMs:F0} ms");
        if (_session is not null && !_faceSupported)
            ImGui.TextWrapped("缺少此角色的面部动画数据，当前只跟随头部。请先用 /posegraph facedump 导出。停止后重新开始生效。");
        ImGui.Separator();
        ImGui.TextWrapped("身体按摄像头姿态重定向，包含可见的髋、膝和腿部；头颈不再叠加角色原先的转头。坐姿需要髋、膝入镜，画面外的腿不能可靠推断。开启同步后按摄像头构图估算角色位移与镜头；不是现实米制位置，也不会自动寻找椅子。停止时恢复。");
        ImGui.TextWrapped("请勿同时播放视频或使用其他姿势编辑器。关闭窗口会停止捕捉并恢复原姿势。");
    }

    private void DrawPreview()
    {
        ImGui.Checkbox("镜像预览（不改变动作映射）", ref _mirrorPreview);
        ImGui.Checkbox("显示识别标记", ref _showTracking);
        if (_preview is { } preview)
        {
            var width = Math.Max(1, Math.Min(ImGui.GetContentRegionAvail().X, 480));
            var size = new Vector2(width, width * preview.Height / Math.Max(1, preview.Width));
            var origin = ImGui.GetCursorScreenPos();
            ImGui.Image(preview.Handle, size,
                _mirrorPreview ? new Vector2(1, 0) : Vector2.Zero,
                _mirrorPreview ? new Vector2(0, 1) : Vector2.One);
            if (_showTracking)
            {
                Vector2 Point(float x, float y) => origin + new Vector2((_mirrorPreview ? 1 - Math.Clamp(x, 0, 1) : Math.Clamp(x, 0, 1)) * size.X, Math.Clamp(y, 0, 1) * size.Y);
                var draw = ImGui.GetWindowDrawList();
                var color = ImGui.GetColorU32(new Vector4(0.2f, 1f, 0.65f, 1f));
                if (_trackingLines is not null)
                    foreach (var line in _trackingLines)
                        if (line.Length == 4) draw.AddLine(Point(line[0], line[1]), Point(line[2], line[3]), color, 2f);
                if (_faceBox is { Length: 4 } box)
                {
                    var a = Point(box[0], box[1]); var b = Point(box[2], box[3]);
                    draw.AddRect(Vector2.Min(a, b), Vector2.Max(a, b), color);
                }
            }
        }
        else ImGui.TextWrapped(_busy ? "等待摄像头画面…" : "开始捕捉后显示实时预览；暂停跟随时画面继续更新。");
        if (_previewError is not null) ImGui.TextWrapped(_previewError);
    }

    private void DrawFaceWeights()
    {
        if (!ImGui.TreeNode("表情权重###liveFaceWeights")) return;
        var changed = false;
        var strength = _config.LiveFaceStrength ?? 1f;
        if (ImGui.SliderFloat("强度##liveFace", ref strength, 0f, 2f, "%.2f")) _config.LiveFaceStrength = strength;
        changed |= ImGui.IsItemDeactivatedAfterEdit();
        if (ImGui.IsItemHovered()) ImGui.SetTooltip("与视频相同：所有表情幅度再乘 0.7；Live 中调整即时生效。");
        foreach (var (key, label, tip) in MainWindow.FaceGroups)
        {
            var value = _config.LiveFaceGains!.TryGetValue(key, out var gain) ? gain : 1f;
            if (ImGui.SliderFloat($"{Loc.T(label)}##liveFace{key}", ref value, 0f, 2f, "%.2f")) _config.LiveFaceGains[key] = value;
            changed |= ImGui.IsItemDeactivatedAfterEdit();
            if (ImGui.IsItemHovered()) ImGui.SetTooltip(Loc.T(tip) + "\n和上面的强度相乘；再大也不会超过游戏里这块能动的范围。");
        }
        if (ImGui.Button("复制视频表情设置"))
        {
            _config.LiveFaceStrength = _config.VideoFaceStrength;
            _config.LiveFaceGains = new(_config.VideoFaceGains);
            _config.LiveMouthLimit = 1f;
            changed = true;
        }
        ImGui.TextDisabled("Live 单独保存，修改不影响视频设置。");
        if (ImGui.TreeNode("额外张嘴限制"))
        {
            var limit = _config.LiveMouthLimit;
            if (ImGui.SliderFloat("张嘴上限", ref limit, 0f, 1f, "%.2f")) _config.LiveMouthLimit = limit;
            changed |= ImGui.IsItemDeactivatedAfterEdit();
            if (ImGui.Button("拉拉菲尔保守预设"))
            {
                _config.LiveFaceGains!["mouth"] = _config.LiveFaceGains["jaw"] = 0.6f;
                _config.LiveMouthLimit = 0.35f;
                changed = true;
            }
            ImGui.TreePop();
        }
        if (changed) _pi.SavePluginConfig(_config);
        ImGui.TreePop();
    }

    private async Task CalibrateAsync()
    {
        if (_calibrating || _session is not { } id || _cancel is not { } cancel) return;
        var token = cancel.Token;
        _calibrating = true;
        _calibrationStarted = System.Diagnostics.Stopwatch.GetTimestamp();
        try
        {
            await Task.Delay(TimeSpan.FromSeconds(2), token);
            if (_disposed || !IsOpen || _session != id || token.IsCancellationRequested) return;
            await _main.Client.CalibrateLiveAsync(id);
            await _framework.RunOnFrameworkThread(() =>
            {
                if (!_disposed && _session == id && !token.IsCancellationRequested) _scene.Recalibrate();
            });
        }
        catch (OperationCanceledException) { }
        catch (Exception ex) { _status = ex.Message; }
        finally { _calibrating = false; }
    }

    private async Task RunAsync()
    {
        _busy = true;
        _paused = false;
        _poseChanged = false;
        _fps = _backendMs = 0;
        _lastFrameTick = 0;
        var cts = _cancel = new CancellationTokenSource();
        var client = _main.Client;
        try
        {
            if (VideoPlaying?.Invoke() == true) throw new InvalidOperationException("请先暂停视频播放。");
            _actor = await _applier.IdentifyAsync(_main.SelectedActor);
            if (_actor is not { ModelReady: true }) throw new InvalidOperationException("请在 GPose 中选择已加载的人物。");
            _backup = await _applier.GetPoseAsync(_actor.Index);
            if (_backup is null) throw new InvalidOperationException("无法读取角色姿势，请检查 Brio。");
            var race = _main.CurrentRace ?? _actor.Race;
            if (race is null) throw new InvalidOperationException("无法识别角色种族。");
            _status = "正在打开摄像头和面捕模型…";
            var camera = _cameras.FirstOrDefault(c => c.Index == _camera);
            var session = await client.StartLiveAsync(_backup, race, _camera, _smoothing, _config.LiveFaceStrength ?? 1f, new(_config.LiveFaceGains!), _config.LiveMouthLimit, camera?.Name, camera?.Path);
            _session = session.Id;
            _faceSupported = session.FaceSupported;
            cts.Token.ThrowIfCancellationRequested();
            _direct.Invalidate();
            var frozen = await _framework.RunOnFrameworkThread(() =>
                _applier.Identify(_actor.Index)?.Signature == _actor.Signature &&
                _applier.Resolve(_actor.Index) is { } a && _brio.Freeze(a));
            if (!frozen) throw new InvalidOperationException("无法冻结目标角色。");
            await _framework.DelayTicks(5);
            if (_matchScene) await _framework.RunOnFrameworkThread(() =>
            {
                if (_applier.Identify(_actor.Index)?.Signature != _actor.Signature) throw new InvalidOperationException("角色已改变。");
                _scene.Begin(_applier.Resolve(_actor.Index)!);
            });
            while (!cts.IsCancellationRequested)
            {
                var frame = await client.LiveFrameAsync(session.Id, cts.Token, _config.LiveFaceStrength ?? 1f, new(_config.LiveFaceGains!), _config.LiveMouthLimit);
                cts.Token.ThrowIfCancellationRequested();
                await UpdatePreviewAsync(frame.Preview, cts.Token);
                await _framework.RunOnFrameworkThread(() =>
                {
                    if (_disposed || cts.IsCancellationRequested) return;
                    var now = System.Diagnostics.Stopwatch.GetTimestamp();
                    if (_lastFrameTick != 0)
                    {
                        var fps = System.Diagnostics.Stopwatch.Frequency / (double)(now - _lastFrameTick);
                        _fps = _fps == 0 ? fps : _fps * 0.8 + fps * 0.2;
                    }
                    _lastFrameTick = now;
                    _backendMs = frame.FrameMs;
                    _trackingLines = frame.Lines;
                    _faceBox = frame.FaceBox;
                    var hands = frame.HandsTracked ?? [];
                    _bodyStatus = $"躯干：{(frame.TorsoTracked ? "跟踪中" : "请让肩和髋入镜")} · 左手：{(hands.Contains("l") ? "跟踪中" : "未匹配")} · 右手：{(hands.Contains("r") ? "跟踪中" : "未匹配")}";
                    if (VideoPlaying?.Invoke() == true) throw new InvalidOperationException("视频开始播放，Live 已停止。");
                    if (_applier.Identify(_actor.Index)?.Signature != _actor.Signature)
                        throw new InvalidOperationException("角色或外观已改变，已停止 Live。");
                    _status = frame.CalibrationMessage ?? (_paused ? "已暂停跟随（摄像头仍开启）"
                        : frame.Pose is not null ? (frame.BodyPose is not null ? "头部与身体跟踪中" : "头部跟踪中；身体未识别")
                        : frame.BodyPose is not null ? "身体跟踪中；人脸未识别" : "未找到可跟踪的人脸或手臂，保持当前姿势。");
                    if (_paused || !frame.Tracked || (frame.Pose is null && frame.BodyPose is null)) return;
                    var pose = frame.BodyPose ?? frame.Pose!;
                    // Full model-space frames must be imported once. Importing
                    // face first, then moving its head ancestor can transform it twice.
                    if (frame.FullPose && frame.BodyPose is not null && frame.Pose is not null)
                    {
                        var merged = JsonNode.Parse(frame.BodyPose)!;
                        var face = JsonNode.Parse(frame.Pose)!;
                        foreach (var bone in face["Bones"]!.AsObject())
                            merged["Bones"]![bone.Key] = bone.Value?.DeepClone();
                        pose = merged.ToJsonString();
                    }
                    using var json = JsonDocument.Parse(pose);
                    var bones = json.RootElement.GetProperty("Bones").EnumerateObject().Select(p => p.Name).ToHashSet(StringComparer.Ordinal);
                    if (frame.BodyPose is not null && frame.Pose is not null)
                    {
                        using var face = JsonDocument.Parse(frame.Pose);
                        bones.UnionWith(face.RootElement.GetProperty("Bones").EnumerateObject().Select(p => p.Name));
                    }
                    var actor = _applier.Resolve(_actor.Index)!;
                    var applied = _direct.Apply(actor, pose, bones, rotationOnly: frame.BodyPose is not null && !frame.FullPose,
                        fullPoseJson: !frame.FullPose && frame.BodyPose is not null ? frame.Pose : null);
                    if (applied is null)
                        throw new InvalidOperationException("当前 Brio 不支持实时写入：" + _direct.Problem);
                    if (applied is true)
                    {
                        _poseChanged = true;
                        if (_matchScene && frame.BodyPose is not null) _scene.Apply(actor, frame.BodyPose, frame.Framing);
                    }
                });
                await Task.Delay(15, cts.Token);
            }
        }
        catch (OperationCanceledException) { _status = "已停止。"; }
        catch (Exception ex) { _status = "Live：" + ex.Message; }
        finally
        {
            // Restore on the game thread, only if this is still the same model.
            try
            {
                if (!_disposed) await _framework.RunOnFrameworkThread(() =>
                {
                    ClearPreview();
                    RestoreScene();
                });
                if (!_disposed && _poseChanged && _actor is { } original && _backup is { } pose)
                    await _framework.RunOnFrameworkThread(() => Restore(original, pose));
            }
            catch (Exception ex) { _status += " 恢复失败：" + ex.Message; }
            if (_session is { } id)
            {
                try { await client.StopLiveAsync(id); }
                catch { _status += " 后端未响应；摄像头会在会话超时后释放。"; }
            }
            _session = null; _actor = null; _backup = null;
            _busy = false; _cancel = null; cts.Dispose();
        }
    }

    private void RestoreScene()
    {
        var same = _actor is { } original && _applier.Identify(original.Index)?.Key == original.Key;
        _scene.Restore(same ? _applier.Resolve(_actor!.Index) : null, _main.InGpose);
    }

    private void Restore(PoseApplier.ActorIdentity original, string pose)
    {
        if (_applier.Identify(original.Index)?.Signature != original.Signature) return;
        if (_applier.Resolve(original.Index) is not { } actor) return;
        using var json = JsonDocument.Parse(pose);
        var bones = json.RootElement.GetProperty("Bones").EnumerateObject().Select(p => p.Name).ToHashSet(StringComparer.Ordinal);
        if (_direct.Apply(actor, pose, bones) is not true) _brio.ApplyPose(actor, pose);
    }

    private async Task UpdatePreviewAsync(string? image, CancellationToken ct)
    {
        if (image is null) return;
        IDalamudTextureWrap? texture = null;
        try
        {
            texture = await _textures.CreateFromImageAsync(Convert.FromBase64String(image));
            await _framework.RunOnFrameworkThread(() =>
            {
                if (_disposed || ct.IsCancellationRequested || !IsOpen) return;
                ClearPreview();
                _preview = texture;
                texture = null;
                _previewError = null;
            });
        }
        catch (Exception ex) { _previewError = "预览加载失败：" + ex.Message; }
        finally { texture?.Dispose(); }
    }

    private void ClearPreview()
    {
        _preview?.Dispose();
        _preview = null;
        _trackingLines = null;
        _faceBox = null;
    }

    public override void OnClose() { _cancel?.Cancel(); ClearPreview(); }
    public void Dispose()
    {
        _disposed = true;
        try { RestoreScene(); } catch { }
        ClearPreview();
        _cancel?.Cancel();
        if (_poseChanged && _actor is { } actor && _backup is { } pose)
        {
            try { Restore(actor, pose); } catch { /* Game may already be shutting down. */ }
        }
    }
}
