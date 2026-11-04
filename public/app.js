const $=s=>document.querySelector(s),ids=['period','duration','depth_ppm','snr','secondary_ppm'];
$('#mb').onclick=()=>document.querySelector('nav').classList.toggle('open');
document.querySelectorAll('.links a').forEach(a=>a.onclick=()=>document.querySelector('nav').classList.remove('open'));
const io=new IntersectionObserver(e=>e.forEach(x=>x.isIntersecting&&x.target.classList.add('in')),{threshold:.1});
document.querySelectorAll('.rv').forEach(e=>io.observe(e));
const tick=()=>$('#clock').textContent=new Date().toLocaleString([], {weekday:'short',month:'short',day:'numeric',hour:'numeric',minute:'2-digit'});tick();setInterval(tick,3e4);
const vals=()=>Object.fromEntries([['kepid',10593626],...ids.map(i=>[i,+$('#'+i).value])]);
function noise(i){return Math.sin(i*12.9898)*43758.5453%1}
function draw(c,v,local){const x=c.getContext('2d'),W=c.width,H=c.height;x.clearRect(0,0,W,H);
 const dp=v.duration/(v.period*24),w=dp/2,sec=v.secondary_ppm/v.depth_ppm,span=local?4*dp:.5,n=400;x.beginPath();
 for(let i=0;i<=n;i++){const ph=-span+2*span*i/n,d=Math.abs(ph);let f=0;
  if(d<w)f=-(d>.8*w?(w-d)/(.2*w):1);const ds=Math.abs(d-.5);if(ds<w)f-=sec;
  f+=noise(i)*.35/Math.sqrt(Math.max(v.snr,1))*3;const px=i/n*W,py=H*.5-f*H*.38;i?x.lineTo(px,py):x.moveTo(px,py)}
 x.strokeStyle='#00d2ff';x.lineWidth=1.6;x.stroke()}
let t;async function run(){const v=vals();ids.forEach((k,i)=>$('#v'+i).textContent=v[k]);draw($('#cg'),v,0);draw($('#cl'),v,1);
 try{const r=await fetch('/api/v1/agent/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(v)});if(r.status==429){$('#vd').textContent='rate limited, slow down';return}
 const d=await r.json();$('#pp').textContent=(d.p_planet*100).toFixed(1)+'%';$('#pb').textContent=(d.baseline*100).toFixed(1)+'%';$('#gg').style.width=d.p_planet*100+'%';$('#vd').textContent=d.final_verdict_report;
 $('#tr').innerHTML=d.trace.map(s=>`<div><b>${s.agent}</b> → ${s.out}</div>`).join('');
 $('#rg').innerHTML=d.rag_context.map(s=>`<div>📖 <b>${s.title}</b>: ${s.note}</div>`).join('');
 if(d.queued_for_hitl)loadQ()}catch(e){$('#vd').textContent='API offline'}}
async function loadQ(){try{const q=await (await fetch('/api/v1/hitl')).json();$('#hq').innerHTML=q.length?q.map(i=>`<div><b>KIC ${i.kepid}</b> · P=${i.p} · ${new Date(i.ts*1e3).toLocaleTimeString()} · awaiting expert label</div>`).join(''):'Queue empty.'}catch(e){}}
ids.forEach(i=>$('#'+i).addEventListener('input',()=>{clearTimeout(t);t=setTimeout(run,250)}));
document.querySelectorAll('.pre button').forEach(b=>b.onclick=()=>{b.dataset.p.split(',').forEach((val,i)=>$('#'+ids[i]).value=val);run()});
$('#cf').onsubmit=async e=>{e.preventDefault();const b=Object.fromEntries(new FormData(e.target));const r=await fetch('/api/v1/contact',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
 $('#cm').textContent=r.ok?'Sent, thank you!':'Please check the fields.';if(r.ok)e.target.reset()};
run();loadQ();

addEventListener('scroll',()=>$('#prog').style.width=scrollY/(document.body.scrollHeight-innerHeight)*100+'%');
const co=new IntersectionObserver(es=>es.forEach(e=>{if(!e.isIntersecting)return;co.unobserve(e.target);const n=+e.target.dataset.n;let k=0;const st=setInterval(()=>{k+=Math.ceil(n/40);if(k>=n){k=n;clearInterval(st)}e.target.textContent=k.toLocaleString()},30)}),{threshold:.5});
document.querySelectorAll('.big').forEach(e=>co.observe(e));
fetch('/api/v1/model').then(r=>r.json()).then(m=>$('#mc').innerHTML=Object.entries(m).map(([k,v])=>`<div><b>${k}</b>: ${typeof v=='object'?JSON.stringify(v):v}</div>`).join('')).catch(()=>$('#mc').textContent='API offline');
let rt;$('#rq').oninput=e=>{clearTimeout(rt);rt=setTimeout(async()=>{const q=e.target.value.trim();if(q.length<2)return;const r=await fetch('/api/v1/rag/search?q='+encodeURIComponent(q));const d=r.ok?await r.json():[];$('#rr').innerHTML=d.length?d.map(x=>`<div>📖 <b>${x.title}</b> (${x.score}): ${x.note}</div>`).join(''):'No matches.'},300)};
document.querySelectorAll('.cp').forEach(b=>b.onclick=()=>{navigator.clipboard.writeText(b.nextElementSibling.textContent.replace('$HOST',location.origin));b.textContent='copied'});

fetch('/api/v1/health').then(r=>r.json()).then(()=>fetch('/api/v1/stats').then(r=>r.json()).then(s=>$('#st').innerHTML='<span style="color:#28c840">●</span> API online · '+(s.analyses_total||0)+' analyses on this instance')).catch(()=>$('#st').textContent='● API offline');
$('#bb').onclick=async()=>{const items=$('#bt').value.trim().split('\n').slice(0,50).map((l,i)=>{const [period,duration,depth_ppm,snr,secondary_ppm]=l.split(',').map(Number);return{kepid:i+1,period,duration,depth_ppm,snr,secondary_ppm:secondary_ppm||0}});
 const r=await fetch('/api/v1/batch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({items})});
 $('#bo').innerHTML=r.ok?(await r.json()).map(x=>`<div>Row ${x.kepid}: <b>${(x.p_planet*100).toFixed(1)}%</b> ${x.p_planet>.65?'candidate':x.p_planet<.4?'false positive':'ambiguous'}</div>`).join(''):'Check the rows (invalid values or rate limit).'};
