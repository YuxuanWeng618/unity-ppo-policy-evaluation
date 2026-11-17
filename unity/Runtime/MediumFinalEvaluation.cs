using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;

// Observation-only wrapper: the course base class still resolves every round.
[DefaultExecutionOrder(-1000)]
public class MediumFinalEvaluation : RollerGameController
{
    public string outputDirectory;
    public string modelLabel;
    public string runId;
    public string modelSha256;
    public int firstSeed = 641600;
    public int targetGames = 10;
    public float evaluationTimeScale = 1f;
    public bool captureFrames = false;
    public int inferenceSeed = 20261030;
    private RenderTexture videoTexture;
    private Texture2D videoReadback;
    private Camera videoCamera;
    [Serializable] public class Round {
        public int episode, spawnSeed, physicsSamples, collisionEntries, attackingContacts;
        public string outcome, reason;
        public float duration, cumulativeReward, pathLength, meanSpeed, nearEdgeFraction, stationaryFraction;
        public float minimumOpponentDistance, spawnX, spawnZ, opponentSpawnX, opponentSpawnZ;
        public float finalX, finalY, finalZ, opponentFinalX, opponentFinalY, opponentFinalZ;
        public float firstContactTime = -1f;
    }
    [Serializable] public class Results {
        public string status, model, runId, modelSha256, startedAt, finishedAt;
        public string actualBehaviorType, actualModel, actualDevice;
        public string behaviorType = "InferenceOnly", inferenceDevice = "CPU", scene = "Assets/Scenes/Medium.unity";
        public int targetGames, games, wins, losses, draws, timeouts, simultaneousFalls;
        public int inferenceSeed, captureFps;
        public string playerColor, opponentColor;
        public float winRate, maxEpisodeTime, timeScale, fixedDeltaTime;
        public int decisionPeriod, opponentHeuristic;
        public string rules = "Original RollerGameController.ResolveRoundIfNeeded; seed resets only spawns; both falls count as draw.";
        public List<Round> rounds = new List<Round>();
    }
    public Results results;
    public bool Done { get; private set; }
    private Round current;
    private float speedSum;
    private int edgeSamples, stationarySamples;
    private Vector3 previousPosition;
    private StreamWriter trajectory, contacts, frameIndex;
    private float nextTrace;
    private int frame;
    private bool ready;
    private static string Num(float f) { return f.ToString("0.000000", CultureInfo.InvariantCulture); }
    private void Awake() {
        Academy.Instance.InferenceSeed = inferenceSeed;
        AgentA.GameController = this; AgentB.GameController = this;
        AgentA.Opponent = AgentB; AgentB.Opponent = AgentA;
        if (playerAgent == null) playerAgent = AgentA;
        Directory.CreateDirectory(outputDirectory);
        Directory.CreateDirectory(Path.Combine(outputDirectory,"frames"));
        trajectory = new StreamWriter(Path.Combine(outputDirectory,"trajectory.csv"),false);
        trajectory.WriteLine("episode,spawn_seed,t,px,py,pz,pvx,pvz,ex,ey,ez,evx,evz,action_x,action_z");
        contacts = new StreamWriter(Path.Combine(outputDirectory,"contacts.csv"),false);
        contacts.WriteLine("episode,t,player_speed_before,opponent_speed_before,attack_advantage");
        frameIndex = new StreamWriter(Path.Combine(outputDirectory,"frames.csv"),false);
        frameIndex.WriteLine("file,episode,t");
        results = new Results {status="running",model=modelLabel,runId=runId,modelSha256=modelSha256,
            startedAt=DateTimeOffset.UtcNow.ToString("o"),targetGames=targetGames,maxEpisodeTime=maxEpisodeTime,
            timeScale=evaluationTimeScale,fixedDeltaTime=Time.fixedDeltaTime,
            decisionPeriod=playerAgent.GetComponent<DecisionRequester>().DecisionPeriod,
            opponentHeuristic=(int)(playerAgent==AgentA?AgentB:AgentA).heuristicMode};
        var actual=playerAgent.GetComponent<BehaviorParameters>();
        results.actualBehaviorType=actual.BehaviorType.ToString();
        results.actualModel=actual.Model!=null?actual.Model.name:"NULL";
        results.actualDevice=actual.InferenceDevice.ToString();
        results.inferenceSeed=inferenceSeed;results.captureFps=captureFrames?30:0;
        results.playerColor=ColorUtility.ToHtmlStringRGB(playerAgent.GetComponent<Renderer>().sharedMaterial.color);
        results.opponentColor=ColorUtility.ToHtmlStringRGB((playerAgent==AgentA?AgentB:AgentA).GetComponent<Renderer>().sharedMaterial.color);
        if(actual.BehaviorType!=BehaviorType.InferenceOnly || actual.Model==null || actual.Model.name!=modelLabel || actual.InferenceDevice!=InferenceDevice.CPU) {
            results.status="failed_runtime_model_validation";Done=true;Save();
            throw new InvalidOperationException("Runtime model binding differs: "+results.actualBehaviorType+" / "+results.actualModel+" / "+results.actualDevice);
        }
        Time.timeScale=evaluationTimeScale;
        Application.runInBackground=true;
        Application.targetFrameRate=captureFrames?30:120;
        Time.captureFramerate=captureFrames?30:0;
        var observer=playerAgent.gameObject.AddComponent<MediumFinalContact>(); observer.owner=this;
        ready=true; Save();
    }
    private void Start() { if(captureFrames) StartCoroutine(Capture()); }
    public override void TryResetEpisode() {
        if(!ready || Done) return;
        if(lastResetStep==Academy.Instance.StepCount) return;
        UnityEngine.Random.InitState(firstSeed+results.rounds.Count);
        base.TryResetEpisode();
        var opponent=playerAgent==AgentA?AgentB:AgentA;
        current=new Round {episode=results.rounds.Count+1,spawnSeed=firstSeed+results.rounds.Count,
            spawnX=playerAgent.transform.localPosition.x,spawnZ=playerAgent.transform.localPosition.z,
            opponentSpawnX=opponent.transform.localPosition.x,opponentSpawnZ=opponent.transform.localPosition.z,
            minimumOpponentDistance=float.MaxValue};
        speedSum=0;edgeSamples=0;stationarySamples=0;
        previousPosition=playerAgent.transform.localPosition;nextTrace=0;
    }
    private void FixedUpdate() {
        if(!ready || Done || current==null || playerAgent.Body==null) return;
        var other=playerAgent==AgentA?AgentB:AgentA;
        Vector3 p=playerAgent.transform.localPosition, e=other.transform.localPosition, v=playerAgent.Body.velocity;
        float speed=new Vector2(v.x,v.z).magnitude;
        current.physicsSamples++; speedSum+=speed;
        if(Mathf.Max(Mathf.Abs(p.x),Mathf.Abs(p.z))/arenaHalfSize>=0.85f)edgeSamples++;
        if(speed<0.2f)stationarySamples++;
        current.pathLength+=new Vector2(p.x-previousPosition.x,p.z-previousPosition.z).magnitude;
        previousPosition=p;
        current.minimumOpponentDistance=Mathf.Min(current.minimumOpponentDistance,new Vector2(p.x-e.x,p.z-e.z).magnitude);
        float t=Time.time-episodeStartTime;
        if(t+0.0001f>=nextTrace) {
            nextTrace=t+0.1f; Vector3 ev=other.Body.velocity;
            var a=playerAgent.GetStoredActionBuffers().DiscreteActions;
            trajectory.WriteLine(string.Join(",",current.episode,current.spawnSeed,Num(t),Num(p.x),Num(p.y),Num(p.z),Num(v.x),Num(v.z),Num(e.x),Num(e.y),Num(e.z),Num(ev.x),Num(ev.z),a.Length>0?a[0]:0,a.Length>1?a[1]:0));
        }
    }
    public void Contact(Collision c) {
        if(Done || current==null) return;
        var opponent=playerAgent==AgentA?AgentB:AgentA;
        if(c.rigidbody!=opponent.Body)return;
        var a=playerAgent.GetComponent<AggressiveCollision>();var b=opponent.GetComponent<AggressiveCollision>();
        float sa=a.velocityBeforePhysicsUpdate.magnitude,sb=b.velocityBeforePhysicsUpdate.magnitude;
        bool advantage=sa>sb+a.speedThreshold;
        current.collisionEntries++; if(advantage)current.attackingContacts++;
        float t=Time.time-episodeStartTime;
        if(current.firstContactTime<0)current.firstContactTime=t;
        contacts.WriteLine(string.Join(",",current.episode,Num(t),Num(sa),Num(sb),advantage?1:0));
    }
    public override void ResolveRoundIfNeeded() {
        if(Done || current==null || roundResolved)return;
        bool af=AgentA.transform.localPosition.y<fallYThreshold,bf=AgentB.transform.localPosition.y<fallYThreshold;
        bool timeout=maxEpisodeTime>0 && Time.time-episodeStartTime>=maxEpisodeTime;
        if(!af && !bf && !timeout)return;
        var completed=current;
        bool selfFell=playerAgent==AgentA?af:bf,opponentFell=playerAgent==AgentA?bf:af;
        var other=playerAgent==AgentA?AgentB:AgentA;
        Vector3 p=playerAgent.transform.localPosition,e=other.transform.localPosition;
        completed.outcome=!selfFell&&opponentFell?"win":selfFell&&!opponentFell?"loss":"draw";
        completed.reason=af&&bf?"both_fell":!af&&!bf?"timeout":selfFell?"player_fell":"opponent_fell";
        completed.duration=Time.time-episodeStartTime;
        completed.cumulativeReward=playerAgent.GetCumulativeReward()+(completed.outcome=="win"?1:completed.outcome=="loss"?-1:0);
        completed.meanSpeed=speedSum/Mathf.Max(1,completed.physicsSamples);
        completed.nearEdgeFraction=(float)edgeSamples/Mathf.Max(1,completed.physicsSamples);
        completed.stationaryFraction=(float)stationarySamples/Mathf.Max(1,completed.physicsSamples);
        completed.finalX=p.x;completed.finalY=p.y;completed.finalZ=p.z;
        completed.opponentFinalX=e.x;completed.opponentFinalY=e.y;completed.opponentFinalZ=e.z;
        results.rounds.Add(completed);results.games=results.rounds.Count;
        if(completed.outcome=="win")results.wins++;
        else if(completed.outcome=="loss")results.losses++;
        else results.draws++;
        if(completed.reason=="timeout")results.timeouts++;
        if(completed.reason=="both_fell")results.simultaneousFalls++;
        results.winRate=(float)results.wins/results.games;
        if(results.games>=targetGames)Done=true;
        // Preserve the course's rewards, counters and episode-ending logic.
        base.ResolveRoundIfNeeded();
        trajectory.Flush();contacts.Flush();frameIndex.Flush();
        if(Done){results.status="completed";results.finishedAt=DateTimeOffset.UtcNow.ToString("o");}
        Save();
        Debug.Log("MEDIUM_EVAL_ROUND "+modelLabel+" "+completed.episode+" "+completed.outcome+" "+completed.reason);
    }
    private IEnumerator Capture() {
        videoCamera=Camera.main;
        if(videoCamera==null)throw new Exception("No game camera found.");
        videoTexture=new RenderTexture(1280,720,24,RenderTextureFormat.ARGB32);
        videoTexture.Create();videoCamera.targetTexture=videoTexture;
        videoReadback=new Texture2D(1280,720,TextureFormat.RGB24,false);
        while(true) {
            yield return new WaitForEndOfFrame();
            if(current!=null) {
                var old=RenderTexture.active;RenderTexture.active=videoTexture;
                videoReadback.ReadPixels(new Rect(0,0,1280,720),0,0);videoReadback.Apply();RenderTexture.active=old;
                string name="frame-"+(frame++).ToString("D6")+".png";
                File.WriteAllBytes(Path.Combine(outputDirectory,"frames",name),videoReadback.EncodeToPNG());
                frameIndex.WriteLine(name+","+current.episode+","+Num(Time.time-episodeStartTime));frameIndex.Flush();
            }
            if(Done)break;
        }
    }
    private void Save(){File.WriteAllText(Path.Combine(outputDirectory,"results.json"),JsonUtility.ToJson(results,true));}
    private void OnGUI(){if(videoTexture!=null)GUI.DrawTexture(new Rect(0,0,Screen.width,Screen.height),videoTexture,ScaleMode.ScaleToFit);if(results!=null)GUI.Label(new Rect(15,15,750,30),modelLabel+" | "+results.games+" / "+targetGames+" | W "+results.wins+" L "+results.losses+" D "+results.draws+" | INFERENCE ONLY");}
    private void OnDestroy(){Time.captureFramerate=0;if(videoCamera!=null)videoCamera.targetTexture=null;if(videoTexture!=null){videoTexture.Release();Destroy(videoTexture);}if(videoReadback!=null)Destroy(videoReadback);if(trajectory!=null)trajectory.Dispose();if(contacts!=null)contacts.Dispose();if(frameIndex!=null)frameIndex.Dispose();}
}
public class MediumFinalContact:MonoBehaviour {
    public MediumFinalEvaluation owner;
    private void OnCollisionEnter(Collision c){if(owner!=null)owner.Contact(c);}
}
