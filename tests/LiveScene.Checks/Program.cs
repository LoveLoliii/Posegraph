using System.Numerics;
using Posegraph;
var brio = new BrioBridge();
var scene = new LiveScene(brio);
var actor = new FakeActor();
const string pose = """{"Bones":{"j_ude_a_l":{"Position":"0.2, 1, 0"},"j_ude_a_r":{"Position":"-0.2, 1, 0"}}}""";
scene.Begin(actor);
for (var i=0;i<14;i++) scene.Apply(actor, pose, [.5f,.5f,.25f,1.33f]);
Check(brio.Writes==0,"must wait for stable samples");
scene.Apply(actor,pose,[.5f,.5f,.25f,1.33f]);
Check(brio.Writes==1 && GposeCamera.Placements==1,"place once after calibration");
for(var i=0;i<100;i++) scene.Apply(actor,pose,[i%2==0?.2f:.8f,.4f,i%2==0?.1f:.7f,1.33f]);
Check(brio.Writes==1 && GposeCamera.Placements==1,"noise must not change size or position after lock");
Check(GposeCamera.Steers==12,"camera convergence must stop");
scene.Recalibrate();
for(var i=0;i<15;i++) scene.Apply(actor,pose,[.5f,.5f,.3f,1.33f]);
Check(brio.Writes==2 && GposeCamera.Placements==2,"explicit recalibration places again");
scene.Restore(actor,true);
Check(brio.LastPosition==Vector3.Zero && GposeCamera.Restores==1,"restore original placement and camera");
Console.WriteLine("LiveScene checks passed");
static void Check(bool value,string message){if(!value)throw new Exception(message);}
class FakeActor: Dalamud.Game.ClientState.Objects.Types.IGameObject {}
namespace Dalamud.Game.ClientState.Objects.Types { public interface IGameObject {} }
namespace Posegraph {
 public class BrioBridge {
  public int Writes; public Vector3 LastPosition;
  public (Vector3?,Quaternion?,Vector3?) GetModelTransform(Dalamud.Game.ClientState.Objects.Types.IGameObject a)=>(Vector3.Zero,Quaternion.Identity,Vector3.One);
  public bool SetModelTransform(Dalamud.Game.ClientState.Objects.Types.IGameObject a,Vector3 p,Quaternion q){Writes++;LastPosition=p;return true;}
 }
 public static class GposeCamera {
  public record SavedSettings;
  public static int Placements,Steers,Restores;
  public static (float FovY,float Aspect,float Zoom)? Projection()=>(1f,1.77f,0f);
  public static SavedSettings SaveSettings()=>new();
  public static void RestoreSettings(SavedSettings s){Restores++;}
  public static bool PlacePosition(Vector3 p,float f){Placements++;return true;}
  public static float SteerView(Vector3 f,Vector3 u){Steers++;return 0;}
 }
}
