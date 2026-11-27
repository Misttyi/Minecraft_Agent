"""HTML 渲染：单局回放播放器、训练看板、任务得分表。全部是自包含的单个 HTML 文件。

- 回放：pov 帧逐帧播放，右侧是当前宏动作、生命/饥饿/太阳、背包、准星；下方折线图与帧同步，
  里程碑在时间轴上打点，旁边是该任务的得分表（达成的打勾）。
- 看板：多组参数的学习曲线（平均任务分、各任务成功率小多图）、课程采样比例、评测表、回放链接。
"""

import base64
import glob
import io
import json
import os
import struct
import zlib

import numpy as np

from mc_rl.tasks import ALL_TASK_NAMES, DEATH_CAUSE_TEXT, TASK_NAMES, get_task, score_series


def save_replay(run_dir, tag, replay):
    folder = os.path.join(run_dir, "replays")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{tag}.npz")
    np.savez_compressed(path, frames=replay["frames"],
                        meta=json.dumps({"trace": replay["trace"], "episode": replay["episode"],
                                         "seed": replay.get("seed")}, ensure_ascii=False))
    return path


# ---------------------------------------------------------------- 图像编码


def _png(frame):
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    h, w, _ = frame.shape
    raw = b"".join(b"\x00" + frame[row].tobytes() for row in range(h))

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def encode_frame(frame):
    try:
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buffer, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    except ImportError:
        return "data:image/png;base64," + base64.b64encode(_png(frame)).decode()


# ---------------------------------------------------------------- 公共样式

STYLE = """
.viz-root{color-scheme:light;--surface-0:#f4f3ef;--surface-1:#fcfcfb;--border:#dcdad3;
--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#7a7873;--grid:#e8e6e0;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#6250d6;--s8:#e34948;
--good:#1f8a4c;--bad:#c23b3a;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
--surface-0:#121211;--surface-1:#1a1a19;--border:#34332f;--text-primary:#f0efec;--text-secondary:#c3c2b7;
--text-muted:#8f8d86;--grid:#2a2a27;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;
--s6:#008300;--s7:#9085e9;--s8:#e66767;--good:#3fb46e;--bad:#e66767;--accent:#3987e5}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-0:#121211;--surface-1:#1a1a19;--border:#34332f;
--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#8f8d86;--grid:#2a2a27;--s1:#3987e5;--s2:#d95926;
--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;--good:#3fb46e;--bad:#e66767;--accent:#3987e5}
*{box-sizing:border-box}body{margin:0}
.viz-root{background:var(--surface-0);color:var(--text-primary);min-height:100vh;
font-family:system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;font-size:14px;line-height:1.55}
.wrap{max-width:1240px;margin:0 auto;padding:20px 24px 60px}
h1{font-size:22px;margin:4px 0 2px}h2{font-size:17px;margin:28px 0 10px}h3{font-size:14px;margin:0 0 6px}
.sub{color:var(--text-secondary)}.muted{color:var(--text-muted)}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:14px 16px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--grid)}
th{color:var(--text-secondary);font-weight:600}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.tag{display:inline-block;padding:0 6px;border-radius:4px;border:1px solid var(--border);font-size:12px;color:var(--text-secondary)}
.ok{color:var(--good);font-weight:600}.no{color:var(--text-muted)}.bad{color:var(--bad);font-weight:600}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:12px 0}
.stat{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:10px 12px}
.stat b{display:block;font-size:22px;font-variant-numeric:tabular-nums}.stat span{font-size:12px;color:var(--text-secondary)}
svg text{fill:var(--text-secondary);font-size:11px}
.tip{position:fixed;pointer-events:none;background:var(--surface-1);border:1px solid var(--border);border-radius:6px;
padding:6px 9px;font-size:12px;box-shadow:0 4px 14px #0002;display:none;z-index:9;min-width:120px}
.tip .row{display:flex;align-items:center;gap:6px}.tip .key{width:12px;height:2px;display:inline-block}
.tip strong{font-variant-numeric:tabular-nums}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:var(--text-secondary);margin:4px 0 6px}
.legend i{display:inline-block;width:14px;height:2px;vertical-align:middle;margin-right:5px}
button,select,input[type=range]{font:inherit}button{background:var(--surface-1);color:var(--text-primary);
border:1px solid var(--border);border-radius:6px;padding:4px 12px;cursor:pointer}
.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
details summary{cursor:pointer;color:var(--text-secondary)}
a{color:var(--accent)}
"""

CHART_JS = r"""
const SERIES=['--s1','--s2','--s3','--s4','--s5','--s6','--s7','--s8'];
function cssVar(name){return getComputedStyle(document.querySelector('.viz-root')).getPropertyValue(name).trim();}
function el(tag,attrs,parent){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);
 for(const k in attrs)e.setAttribute(k,attrs[k]);if(parent)parent.appendChild(e);return e;}
function niceMax(v){if(v<=0)return 1;const p=Math.pow(10,Math.floor(Math.log10(v)));const m=v/p;
 return (m<=1?1:m<=2?2:m<=5?5:10)*p;}
function fmt(v){if(v==null||isNaN(v))return '—';const a=Math.abs(v);
 return a>=1e6?(v/1e6).toFixed(2)+'M':a>=1e4?(v/1e3).toFixed(0)+'k':a>=100?v.toFixed(0):a>=10?v.toFixed(1):v.toFixed(2);}
const tip=document.createElement('div');tip.className='tip';document.addEventListener('DOMContentLoaded',()=>document.querySelector('.viz-root').appendChild(tip));
function showTip(ev,title,rows){tip.textContent='';const t=document.createElement('div');t.className='muted';t.textContent=title;tip.appendChild(t);
 rows.forEach(r=>{const d=document.createElement('div');d.className='row';const k=document.createElement('span');k.className='key';k.style.background=r.color;
 const s=document.createElement('strong');s.textContent=r.value;const l=document.createElement('span');l.className='muted';l.textContent=r.label;
 d.append(k,s,l);tip.appendChild(d);});tip.style.display='block';
 const x=Math.min(ev.clientX+14,window.innerWidth-tip.offsetWidth-8);tip.style.left=x+'px';tip.style.top=(ev.clientY+12)+'px';}
function hideTip(){tip.style.display='none';}
/* 折线图：series=[{name,color,points:[[x,y],...]}]，单一 y 轴；opts={xLabel,yMax,yMin,height,markers:[{x,label}],onHover} */
function lineChart(host,series,opts){opts=opts||{};host.textContent='';const W=Math.max(Math.round(host.getBoundingClientRect().width),240)||520,H=opts.height||200;
 const m={l:44,r:opts.right||70,t:10,b:28};const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,width:'100%',preserveAspectRatio:'xMinYMin meet',role:'img'},host);svg.style.display='block';svg.style.maxWidth='100%';
 let xs=[],ys=[];series.forEach(s=>s.points.forEach(p=>{xs.push(p[0]);if(p[1]!=null)ys.push(p[1]);}));
 if(!xs.length){const t=el('text',{x:10,y:20},svg);t.textContent='暂无数据';return;}
 const x0=opts.xMin!=null?opts.xMin:Math.min(...xs),x1=Math.max(...xs,x0+1e-9);
 const y0=opts.yMin!=null?opts.yMin:Math.min(0,...ys),y1=opts.yMax!=null?opts.yMax:niceMax(Math.max(...ys,1e-9));
 const X=v=>m.l+(v-x0)/(x1-x0||1)*(W-m.l-m.r),Y=v=>H-m.b-(v-y0)/(y1-y0||1)*(H-m.t-m.b);
 for(let i=0;i<=4;i++){const v=y0+(y1-y0)*i/4;el('line',{x1:m.l,x2:W-m.r,y1:Y(v),y2:Y(v),stroke:cssVar('--grid')},svg);
  const t=el('text',{x:m.l-6,y:Y(v)+4,'text-anchor':'end'},svg);t.textContent=fmt(v);}
 for(let i=0;i<=4;i++){const v=x0+(x1-x0)*i/4;const t=el('text',{x:X(v),y:H-8,'text-anchor':'middle'},svg);t.textContent=fmt(v);}
 if(opts.xLabel){const t=el('text',{x:W-m.r,y:H-8,'text-anchor':'start',dx:22},svg);t.textContent=opts.xLabel;}
 (opts.markers||[]).forEach(mk=>{el('line',{x1:X(mk.x),x2:X(mk.x),y1:m.t,y2:H-m.b,stroke:cssVar('--text-muted'),'stroke-dasharray':'2 3'},svg);});
 const ends=[];series.forEach(s=>{const pts=s.points.filter(p=>p[1]!=null);if(!pts.length)return;
  el('path',{d:pts.map((p,i)=>(i?'L':'M')+X(p[0]).toFixed(1)+' '+Y(p[1]).toFixed(1)).join(''),fill:'none',stroke:s.color,'stroke-width':2,'stroke-linejoin':'round'},svg);
  if(opts.directLabels!==false&&series.length<=4){const p=pts[pts.length-1];ends.push({x:X(p[0])+6,y:Y(p[1])+4,name:s.name});}});
 ends.sort((a,b)=>a.y-b.y);for(let k=1;k<ends.length;k++)if(ends[k].y-ends[k-1].y<13)ends[k].y=ends[k-1].y+13;
 ends.forEach(e=>{const t=el('text',{x:e.x,y:e.y},svg);t.textContent=e.name;});
 const cross=el('line',{y1:m.t,y2:H-m.b,stroke:cssVar('--text-muted'),visibility:'hidden'},svg);
 const hit=el('rect',{x:m.l,y:m.t,width:W-m.l-m.r,height:H-m.t-m.b,fill:'transparent'},svg);
 const toX=ev=>{const r=svg.getBoundingClientRect();return x0+((ev.clientX-r.left)*W/r.width-m.l)/(W-m.l-m.r)*(x1-x0);};
 hit.addEventListener('pointermove',ev=>{const xv=toX(ev);
  const rows=[];let snap=null;series.forEach(s=>{let best=null;s.points.forEach(p=>{if(best==null||Math.abs(p[0]-xv)<Math.abs(best[0]-xv))best=p;});
   if(best){snap=snap==null?best[0]:snap;rows.push({color:s.color,value:fmt(best[1]),label:s.name});}});
  if(snap==null)return;cross.setAttribute('x1',X(snap));cross.setAttribute('x2',X(snap));cross.setAttribute('visibility','visible');
  showTip(ev,(opts.xLabel||'x')+' '+fmt(snap),rows);if(opts.onHover)opts.onHover(snap);});
 hit.addEventListener('pointerleave',()=>{cross.setAttribute('visibility','hidden');hideTip();});
 hit.addEventListener('click',ev=>{if(opts.onClick)opts.onClick(toX(ev));});
 return {svg,X,Y,setCursor:v=>{cross.setAttribute('x1',X(v));cross.setAttribute('x2',X(v));cross.setAttribute('visibility','visible');}};}
function legend(host,series){host.textContent='';host.className='legend';series.forEach(s=>{const sp=document.createElement('span');
 const i=document.createElement('i');i.style.background=s.color;sp.append(i,document.createTextNode(s.name));host.appendChild(sp);});}
"""


def _page(title, body, data, script):
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{STYLE}</style></head>
<body><div class="viz-root"><div class="wrap">{body}</div></div>
<script>const DATA={payload};</script>
<script>{CHART_JS}</script>
<script>{script}</script>
</body></html>"""


# ---------------------------------------------------------------- 得分表 HTML


def score_tables_html_fragment():
    series = score_series()
    stage_rows = "".join(
        f"<tr><td class='num'>{st}</td><td>{title}</td><td>{text}</td><td>"
        + ", ".join(f"<code>{t['name']}</code>" for t in series["tasks"] if t["stage"] == st) + "</td></tr>"
        for st, title, text in series["stages"])
    parts = ["<div class='card' style='margin-bottom:14px'><h3>课程阶段</h3>"
             f"<div class='muted'>前沿阶段成功率均值 ≥ {series['unlock']['success']}、得分均值 ≥ {series['unlock']['score']:g} "
             f"或累计 ≥ {series['unlock']['episodes']} 局解锁下一阶段；"
             "早期阶段保留采样。</div><table><tr><th class='num'>阶段</th><th>名称</th><th>内容</th><th>任务</th></tr>"
             f"{stage_rows}</table></div>", '<div class="grid2">']
    series["tasks"] = sorted(series["tasks"], key=lambda t: t["stage"])
    for index, task in enumerate(series["tasks"]):
        rows = "".join(
            f"<tr><td><code>{m['key']}</code></td><td>{m['text']}</td><td class='num'>{m['points']:g}</td>"
            f"<td class='num'>{m['reward']:g}</td><td class='muted'>{', '.join(m['requires']) or '—'}"
            f"{' · 局末' if m['at_end'] else ''}</td></tr>"
            for m in task["milestones"]
        )
        dense = "，".join(f"{k} {v:g}" for k, v in task["dense"].items())
        badges = " ".join(f"<span class='tag'>★ {a['text']}</span>" for a in task["achievements"])
        parts.append(
            f"<div class='card'><h3>{task['title']} <span class='tag'>{task['name']}</span>"
            f"{' <span class=tag>扩展</span>' if task['extension'] else ''}</h3>"
            f"<div class='sub'>{task['goal']}</div><div class='muted'>阶段 {task['stage']} · 时长 {task['horizon']} 刻 · 成功 = "
            f"<code>{task['success']}</code> · 采样权重 {task['weight']}</div>"
            f"<table><tr><th>里程碑</th><th>说明</th><th class='num'>得分</th><th class='num'>奖励</th><th>前置</th></tr>"
            f"{rows}</table><div class='muted' style='margin-top:6px'>稠密塑形：{dense or '—'}</div>"
            f"{('<div style=margin-top:6px>成就：' + badges + '</div>') if badges else ''}</div>"
        )
    parts.append("</div>")
    item_rows = "".join(f"<tr><td><code>{k}</code></td><td class='num'>{v}</td></tr>" for k, v in series["item_rewards"])
    parts.append(
        "<div class='card' style='margin-top:14px'><h3>MineRL 物品奖励（每种一次）</h3>"
        f"<table><tr><th>物品</th><th class='num'>奖励</th></tr>{item_rows}</table>"
        "<div class='muted' style='margin-top:6px'>最终评分口径见作业说明；此处只列训练奖励。</div></div>"
    )
    return "".join(parts)


def write_score_tables_html(path):
    body = ("<h1>任务系列得分表</h1><div class='sub'>每个任务的里程碑得分合计 100；训练奖励单独设定。"
            "里程碑按前置条件依次达成。</div>" + score_tables_html_fragment())
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_page("任务系列得分表", body, {}, ""))
    return path


# ---------------------------------------------------------------- 回放

REPLAY_SCRIPT = r"""
const T=DATA.trace,F=DATA.frames;let i=0,playing=false,timer=null;
const img=document.getElementById('frame'),slider=document.getElementById('slider');slider.max=F.length-1;
const lines=[{key:'life',name:'生命',color:cssVar('--s8')},{key:'food',name:'饥饿',color:cssVar('--s4')},
 {key:'sunx20',name:'太阳×20',color:cssVar('--s1')},{key:'score',name:'任务分÷5',color:cssVar('--s3')}];
const pts=k=>T.map(r=>[r.ticks,k==='sunx20'?r.sun*20:k==='score'?r.score/5:r[k]]);
const series=lines.map(l=>({name:l.name,color:l.color,points:pts(l.key)}));
legend(document.getElementById('legend'),series);
const marks=[];T.forEach(r=>(r.newly||[]).forEach(k=>marks.push({x:r.ticks,label:k})));
function nearestIndex(tick){let b=0;T.forEach((r,j)=>{if(Math.abs(r.ticks-tick)<Math.abs(T[b].ticks-tick))b=j;});return b;}
const chart=lineChart(document.getElementById('chart'),series,{xLabel:'刻',yMax:20,yMin:0,height:220,markers:marks,
 onClick:t=>show(nearestIndex(t))});
const done={};
function show(j){i=Math.max(0,Math.min(F.length-1,j));const r=T[i];img.src=F[i];slider.value=i;
 chart.setCursor(r.ticks);
 const set=(id,v)=>{document.getElementById(id).textContent=v;};
 set('tick',r.ticks);set('macro',r.macro);set('life',fmt(r.life));set('food',fmt(r.food));set('sun',fmt(r.sun));
 set('dy',fmt(r.dy));set('score',fmt(r.score));set('reward',fmt(r.reward));set('hand',r.hand);set('ray',r.ray||'—');
 set('near',[r.nearest_log!=null?'树 '+fmt(r.nearest_log):null,r.nearest_animal!=null?'动物 '+fmt(r.nearest_animal):null,
   r.nearest_hostile!=null?'怪物 '+fmt(r.nearest_hostile):null].filter(Boolean).join(' · ')||'—');
 set('inv',['log','planks','stick','crafting_table','wooden_sword','wooden_pickaxe','dirt','meat','coal','torch']
   .filter(k=>r[k]>0).map(k=>k+'×'+r[k]).join('  ')||'空');
 const got=new Set();for(let j=0;j<=i;j++)(T[j].newly||[]).forEach(k=>got.add(k));
 document.querySelectorAll('[data-ms]').forEach(td=>{const ok=got.has(td.dataset.ms);td.textContent=ok?'✓':'·';td.className=ok?'ok':'no';});
 const log=document.getElementById('events');log.textContent='';
 for(let j=Math.max(0,i-12);j<=i;j++){const d=document.createElement('div');const r2=T[j];
  d.textContent=r2.ticks+'  '+r2.macro+(r2.newly&&r2.newly.length?'  ★ '+r2.newly.join(','):'')+(r2.reward?'  ('+fmt(r2.reward)+')':'');
  if(j===i)d.style.fontWeight='600';log.appendChild(d);}}
slider.addEventListener('input',()=>show(+slider.value));
document.getElementById('play').addEventListener('click',()=>{playing=!playing;document.getElementById('play').textContent=playing?'暂停':'播放';
 clearInterval(timer);if(playing)timer=setInterval(()=>{if(i>=F.length-1){playing=false;clearInterval(timer);return;}show(i+1);},
 1000/(+document.getElementById('speed').value));});
document.getElementById('speed').addEventListener('change',()=>{if(playing){clearInterval(timer);timer=setInterval(()=>show(i+1),1000/(+document.getElementById('speed').value));}});
document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')show(i+1);if(e.key==='ArrowLeft')show(i-1);if(e.key===' '){e.preventDefault();document.getElementById('play').click();}});
show(0);
"""


def render_replay(npz_path, out_path=None, max_frames=1800):
    with np.load(npz_path, allow_pickle=False) as data:
        frames = data["frames"]
        meta = json.loads(str(data["meta"]))
    trace = meta["trace"]
    episode = meta.get("episode") or {}
    keep = list(range(len(trace)))
    if len(keep) > max_frames:
        stride = int(np.ceil(len(keep) / max_frames))
        keep = sorted(set(keep[::stride]) | {j for j, r in enumerate(trace) if r.get("newly")} | {len(trace) - 1})
    trace = [trace[j] for j in keep]
    encoded = [encode_frame(frames[j]) for j in keep]
    task = get_task(episode.get("task", TASK_NAMES[0]))
    rows = "".join(
        f"<tr><td data-ms='{m.key}' class='no'>·</td><td>{m.text}</td><td class='num'>{m.points:g}</td></tr>"
        for m in task.milestones
    )
    cause = episode.get("death_cause")
    status = ("<span class='bad'>死亡：" + DEATH_CAUSE_TEXT.get(cause, cause) + "</span>") if episode.get("died") else \
        ("<span class='ok'>成功</span>" if episode.get("success") else "<span class='muted'>未成功</span>")
    got = episode.get("achievements") or {}
    badges = "".join(
        f"<span class='tag' style='margin-right:6px;{'' if a.key in got else 'opacity:.45'}'>"
        f"{'★' if a.key in got else '☆'} {a.text}</span>" for a in task.achievements)
    body = f"""
<h1>回放 · {task.title} <span class='tag'>{task.name}</span></h1>
<div style='margin:4px 0'>{badges}</div>
<div class='sub'>种子 {meta.get('seed')} · 存活 {episode.get('survival_steps', '—')} 刻 · 任务分 {episode.get('score', 0):g}/100 · {status}</div>
<div class='grid2' style='margin-top:14px;grid-template-columns:minmax(300px,420px) 1fr'>
 <div class='card'><img id='frame' style='width:100%;image-rendering:pixelated;border-radius:4px;background:#000' alt='pov'>
  <div class='toolbar' style='margin-top:8px'><button id='play'>播放</button>
  <input id='slider' type='range' min='0' value='0' style='flex:1'>
  <select id='speed'><option value='4'>4 帧/s</option><option value='10' selected>10 帧/s</option><option value='25'>25 帧/s</option></select></div>
  <div class='muted' style='font-size:12px'>← → 逐帧，空格播放；点击下方曲线跳转。</div></div>
 <div class='card'><div class='stats' style='margin-top:0'>
  <div class='stat'><b id='tick'></b><span>游戏刻</span></div><div class='stat'><b id='macro' style='font-size:16px'></b><span>宏动作</span></div>
  <div class='stat'><b id='life'></b><span>生命</span></div><div class='stat'><b id='food'></b><span>饥饿</span></div>
  <div class='stat'><b id='sun'></b><span>太阳亮度</span></div><div class='stat'><b id='dy'></b><span>相对出生高度</span></div>
  <div class='stat'><b id='score'></b><span>任务分</span></div><div class='stat'><b id='reward'></b><span>本步奖励</span></div></div>
  <table><tr><th>主手</th><td id='hand'></td></tr><tr><th>背包</th><td id='inv'></td></tr>
  <tr><th>准星（特权）</th><td id='ray'></td></tr><tr><th>最近目标（特权）</th><td id='near'></td></tr></table></div>
</div>
<div class='grid2' style='margin-top:14px;grid-template-columns:2fr 1fr'>
 <div class='card'><h3>状态曲线</h3><div id='legend'></div><div id='chart'></div>
  <div class='muted' style='font-size:12px'>虚线 = 拿到里程碑的时刻。</div></div>
 <div class='card'><h3>得分表</h3><table><tr><th></th><th>里程碑</th><th class='num'>分</th></tr>{rows}</table>
  <h3 style='margin-top:12px'>最近动作</h3><div id='events' style='font-family:monospace;font-size:12px'></div></div>
</div>
<details style='margin-top:14px'><summary>本局汇总 JSON</summary><pre style='white-space:pre-wrap;font-size:12px'>{json.dumps(episode, ensure_ascii=False, indent=1)}</pre></details>
"""
    out_path = out_path or os.path.splitext(npz_path)[0] + ".html"
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(_page(f"回放 {task.title}", body, {"trace": trace, "frames": encoded}, REPLAY_SCRIPT))
    return out_path


# ---------------------------------------------------------------- 看板

DASH_SCRIPT = r"""
const runs=DATA.runs,tasks=DATA.tasks;const color=i=>cssVar(SERIES[i%8]);
const runSeries=key=>runs.map((r,i)=>({name:r.name,color:color(i),points:r.metrics.map(m=>[m.ticks,key(m)])}));
legend(document.getElementById('legend-main'),runs.map((r,i)=>({name:r.name,color:color(i)})));
lineChart(document.getElementById('c-score'),runSeries(m=>m.mean_score),{xLabel:'环境刻',yMax:100,yMin:0,height:240});
const sm=document.getElementById('smalls');
const smHosts=tasks.map(t=>{const c=document.createElement('div');c.className='card';const h=document.createElement('h3');h.textContent=t.title+'  成功率';
 const d=document.createElement('div');c.append(h,d);sm.appendChild(c);return [t,d];});
smHosts.forEach(([t,d])=>lineChart(d,runSeries(m=>m.tasks&&m.tasks[t.name]?m.tasks[t.name].success:null),{xLabel:'刻',yMax:1,yMin:0,height:150,right:16,directLabels:false}));
const cur=document.getElementById('curr');
const curHosts=runs.map((r,i)=>{const last=r.metrics[r.metrics.length-1];if(!last||!last.curriculum)return null;const c=document.createElement('div');c.className='card';
 const h=document.createElement('h3');h.textContent=r.name+'  课程采样比例'+(last.curriculum.stage!=null?'（已解锁到阶段 '+last.curriculum.stage+'）':'');const d=document.createElement('div');c.append(h,d);cur.appendChild(c);return [r,i,last,d];}).filter(Boolean);
curHosts.forEach(([r,i,last,d])=>{
 const probs=last.curriculum.probs;const names=Object.keys(probs);const W=Math.max(d.getBoundingClientRect().width,300),rowH=18,H=names.length*rowH+8;
 const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,width:'100%',preserveAspectRatio:'xMinYMin meet'},d);const mx=Math.max(...Object.values(probs),1e-9);
 names.forEach((n,j)=>{const t=el('text',{x:0,y:j*rowH+13},svg);t.textContent=n;const w=(W-210)*probs[n]/mx;
  const b=el('rect',{x:96,y:j*rowH+4,width:Math.max(w,1),height:rowH-6,rx:3,fill:color(i)},svg);
  const v=el('text',{x:100+w,y:j*rowH+13,dx:2},svg);v.textContent=(probs[n]*100).toFixed(0)+'%  成功 '+((last.curriculum.rate[n]||0)*100).toFixed(0)+'%';
  b.addEventListener('pointermove',ev=>showTip(ev,n,[{color:color(i),value:(probs[n]*100).toFixed(1)+'%',label:'采样'}]));b.addEventListener('pointerleave',hideTip);});});
"""


def _read_jsonl(path, limit=None):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows[-limit:] if limit else rows


def render_dashboard(run_dirs, out_path, title="MineRL 生存一天 · 训练看板", eval_files=()):
    runs = []
    for run_dir in run_dirs:
        metrics = _read_jsonl(os.path.join(run_dir, "metrics.jsonl"))
        slim = [{"ticks": m["ticks"], "mean_score": m.get("mean_score", 0), "tasks": m.get("tasks", {}),
                 "curriculum": m.get("curriculum") if j == len(metrics) - 1 else None,
                 "entropy": m.get("entropy"), "bc_acc": m.get("bc_acc")} for j, m in enumerate(metrics)]
        cfg = {}
        cfg_path = os.path.join(run_dir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as handle:
                cfg = json.load(handle)
        replays = sorted(glob.glob(os.path.join(run_dir, "replays", "*.html")))
        runs.append({"name": os.path.basename(os.path.normpath(run_dir)), "metrics": slim, "cfg": cfg,
                     "replays": [os.path.relpath(p, os.path.dirname(out_path)) for p in replays][-12:],
                     "dir": run_dir})
    present = {name for run in runs for m in run["metrics"] for name in (m.get("tasks") or {})}
    shown = [n for n in ALL_TASK_NAMES if n in TASK_NAMES or n in present]
    tasks = [{"name": n, "title": get_task(n).title} for n in shown]
    # 最终表
    head = "".join(f"<th class='num'>{get_task(n).title}</th>" for n in shown)
    body_rows = []
    for run in runs:
        last = run["metrics"][-1] if run["metrics"] else {}
        cells = []
        for name in shown:
            stat = (last.get("tasks") or {}).get(name)
            cells.append(f"<td class='num'>{stat['score']:.0f} / {stat['success'] * 100:.0f}%</td>" if stat else
                         "<td class='num muted'>—</td>")
        cfg = run["cfg"]
        knobs = " ".join(f"{k}={cfg.get(k)}" for k in ("arch", "lr", "ent_coef", "bc_coef", "num_envs") if k in cfg)
        body_rows.append(f"<tr><td><b>{run['name']}</b><div class='muted' style='font-size:11px'>{knobs}</div></td>"
                         f"<td class='num'>{last.get('ticks', 0):,}</td><td class='num'>{last.get('mean_score', 0):.1f}</td>"
                         + "".join(cells) + "</tr>")
    replay_links = "".join(
        f"<div class='card'><h3>{run['name']}</h3>" + ("".join(
            f"<div><a href='{p}'>{os.path.basename(p)}</a></div>" for p in run["replays"]) or "<span class='muted'>暂无</span>")
        + "</div>" for run in runs)
    eval_html = ""
    for path in eval_files:
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        eval_html += _eval_card(report, os.path.basename(path))
    tops = sorted(runs, key=lambda r: -(r["metrics"][-1]["mean_score"] if r["metrics"] else 0))
    best = tops[0] if tops else None
    stat_html = ""
    if best and best["metrics"]:
        last = best["metrics"][-1]
        stat_html = (f"<div class='stats'><div class='stat'><b>{len(runs)}</b><span>并行参数组</span></div>"
                     f"<div class='stat'><b>{best['name']}</b><span>当前最好</span></div>"
                     f"<div class='stat'><b>{last['mean_score']:.1f}</b><span>最好组平均任务分</span></div>"
                     f"<div class='stat'><b>{sum(r['metrics'][-1]['ticks'] for r in runs if r['metrics']):,}</b>"
                     f"<span>累计环境刻</span></div></div>")
    body = f"""<h1>{title}</h1><div class='sub'>PPO + 模仿学习，多任务课程。平均任务分 = 各任务最近 50 局得分表分数的平均（0–100）。</div>
{stat_html}
<h2>学习曲线</h2><div class='card'><div id='legend-main'></div><div id='c-score'></div></div>
<h2>各任务成功率</h2><div class='grid3' id='smalls'></div>
<h2>最新一次更新的任务分 / 成功率</h2><div class='card' style='overflow-x:auto'><table>
<tr><th>参数组</th><th class='num'>环境刻</th><th class='num'>平均分</th>{head}</tr>{''.join(body_rows)}</table></div>
{('<h2>评测</h2>' + eval_html) if eval_html else ''}
<h2>课程</h2><div class='grid3' id='curr'></div>
<h2>回放</h2><div class='grid3'>{replay_links}</div>
<h2>任务系列得分表</h2>{score_tables_html_fragment()}"""
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(_page(title, body, {"runs": runs, "tasks": tasks}, DASH_SCRIPT))
    return out_path


def _eval_card(report, name):
    rows = "".join(
        f"<tr><td class='num'>{e['seed']}</td><td class='num'>{e['survival_steps']}</td>"
        f"<td>{'<span class=ok>是</span>' if e['survived_day'] else '否'}</td><td class='num'>{e.get('life')}</td>"
        f"<td class='num'>{e.get('food')}</td><td>{DEATH_CAUSE_TEXT.get(e.get('death_cause'), e.get('death_cause') or '—')}</td></tr>"
        for e in report.get("episodes", []))
    causes = report.get("death_causes") or {}
    cause_text = "，".join(f"{DEATH_CAUSE_TEXT.get(k, k)} {v}" for k, v in causes.items()) or "—"
    return (f"<div class='card' style='margin-bottom:12px'><h3>{name}</h3><div class='stats'>"
            f"<div class='stat'><b>{report.get('median_survival_steps')}</b><span>中位存活步数</span></div>"
            f"<div class='stat'><b>{report.get('survived_day_count')}/{len(report.get('episodes', []))}</b><span>活过一天</span></div>"
            f"<div class='stat'><b style='font-size:15px'>{cause_text}</b><span>死因分布</span></div></div>"
            f"<table><tr><th class='num'>种子</th><th class='num'>存活步数</th><th>活过一天</th><th class='num'>生命</th>"
            f"<th class='num'>饥饿</th><th>死因</th></tr>{rows}</table></div>")


def render_all_replays(run_dir):
    out = []
    for path in sorted(glob.glob(os.path.join(run_dir, "replays", "*.npz"))):
        html = os.path.splitext(path)[0] + ".html"
        if not os.path.exists(html) or os.path.getmtime(html) < os.path.getmtime(path):
            out.append(render_replay(path, html))
    return out
