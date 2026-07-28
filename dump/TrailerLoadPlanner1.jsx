import { useState, useMemo, useEffect, useCallback } from "react";

// ╔══════════════════════════════════════════════════════════════════╗
// ║  TRAILER LOAD PLANNER v6 — True Shape Footprints               ║
// ║  Measurement-based 3D trailer · Cutout-refined item shapes     ║
// ║  Mask-aware placement with all rotations · Live weight heatmap ║
// ║  Eq30 sigmoid · Variance Suppression · 5-phase Tetris          ║
// ╚══════════════════════════════════════════════════════════════════╝

const GRID=2,HEAT=4;
const WT={extra:{label:"Extra Heavy",lbs:100,color:"#991b1b",pri:4,desc:"Appliances, solid wood"},heavy:{label:"Heavy",lbs:60,color:"#c2410c",pri:3,desc:"Dressers, tables"},medium:{label:"Medium",lbs:30,color:"#a16207",pri:2,desc:"Packed boxes"},light:{label:"Light",lbs:10,color:"#15803d",pri:1,desc:"Clothes, pillows"}};
const WT_KEYS=["extra","heavy","medium","light"];
const CAT={furniture:{label:"Furniture",emoji:"🛋️",phase:1,floorOnly:true,canBearWeight:true,isGapFiller:false,defaultWt:"heavy",desc:"Couch, table, dresser"},appliance:{label:"Appliance",emoji:"🔌",phase:1,floorOnly:true,canBearWeight:false,isGapFiller:false,defaultWt:"extra",desc:"Washer, dryer, fridge"},box:{label:"Box",emoji:"📦",phase:2,floorOnly:false,canBearWeight:true,isGapFiller:false,defaultWt:"medium",desc:"Moving boxes, bins"},fragile:{label:"Fragile",emoji:"⚠️",phase:3,floorOnly:false,canBearWeight:false,isGapFiller:false,defaultWt:"medium",desc:"TV, mirrors, electronics"},long:{label:"Long Item",emoji:"📏",phase:4,floorOnly:false,canBearWeight:false,isGapFiller:false,defaultWt:"light",desc:"Lamps, rods, brooms"},soft:{label:"Soft/Loose",emoji:"🛏️",phase:5,floorOnly:false,canBearWeight:false,isGapFiller:true,defaultWt:"light",desc:"Pillows, blankets, bags"}};
const CAT_KEYS=["furniture","appliance","box","fragile","long","soft"];
const parseHex=h=>[parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)];
function lerpColor(a,b,t){const[ar,ag,ab]=parseHex(a),[br,bg,bb]=parseHex(b);return`rgb(${Math.round(ar+(br-ar)*t)},${Math.round(ag+(bg-ag)*t)},${Math.round(ab+(bb-ab)*t)})`;}
const HS=[[0,"#3b82f6"],[.25,"#22d3ee"],[.5,"#eab308"],[.75,"#f97316"],[1,"#dc2626"]];
function heatColor(v,mx){if(v<=0||mx<=0)return"transparent";const t=Math.min(v/mx,1);for(let i=0;i<HS.length-1;i++){if(t<=HS[i+1][0]){const u=(t-HS[i][0])/(HS[i+1][0]-HS[i][0]);return lerpColor(HS[i][1],HS[i+1][1],u);}}return HS[HS.length-1][1];}
const CC={furniture:"#7c3aed",appliance:"#0891b2",box:"#2563eb",fragile:"#e11d48",long:"#65a30d",soft:"#d97706"};
const toIn=(ft,inc)=>(parseFloat(ft)||0)*12+(parseFloat(inc)||0);
const fmtD=i=>{if(!i||i<=0)return'0"';const f=Math.floor(i/12),r=Math.round(i%12);return f>0?(r>0?`${f}'${r}"`:`${f}'`):`${r}"`;};
const fmtD3=(l,w,h)=>`${fmtD(l)} × ${fmtD(w)} × ${fmtD(h)}`;
let _id=1;const nid=()=>_id++;
const itemWeight=(it)=>it.exactLbs!=null?it.exactLbs:WT[it.weight].lbs;
const shade=(hex,f)=>{const[r,g,b]=parseHex(hex);return`rgb(${Math.round(r*f)},${Math.round(g*f)},${Math.round(b*f)})`;};
function interp(pts,val,vK,wK){if(!pts.length)return 0;const s=[...pts].sort((a,b)=>a[vK]-b[vK]);if(val<=s[0][vK])return s[0][wK];if(val>=s[s.length-1][vK])return s[s.length-1][wK];for(let i=0;i<s.length-1;i++){if(val>=s[i][vK]&&val<=s[i+1][vK]){const t=(val-s[i][vK])/(s[i+1][vK]-s[i][vK]);return s[i][wK]+t*(s[i+1][wK]-s[i][wK]);}}return s[s.length-1][wK];}

// ╔══════════════════════════════════════════════════════════════════╗
// ║  FOOTPRINT SYSTEM — True shapes from measurements + cutouts    ║
// ╚══════════════════════════════════════════════════════════════════╝

function buildFootprint(item){
  const gw=Math.ceil(item.w/GRID),gl=Math.ceil(item.l/GRID);
  if(gw<=0||gl<=0)return{mask:[],gw:0,gl:0};
  const mask=Array.from({length:gl},()=>new Uint8Array(gw).fill(1));

  // Round/cylinder: inscribe circle in bounding box
  if(item.surface==="round"||item.surface==="cylinder"){
    const rad=Math.min(gw,gl)/2;
    const cx=gw/2,cy=gl/2;
    for(let y=0;y<gl;y++)for(let x=0;x<gw;x++){
      const dx2=(x+.5-cx),dy2=(y+.5-cy);
      if(dx2*dx2+dy2*dy2>rad*rad)mask[y][x]=0;}
  }

  // Apply cutouts (only for non-round items)
  if(item.surface!=="round"&&item.surface!=="cylinder"){
    for(const c of(item.cutouts||[])){
      const cw=Math.ceil(c.cw/GRID),cl=Math.ceil(c.cl/GRID);
      let sx=0,sy=0;
      if(c.corner==="fr"||c.corner==="br")sx=gw-cw;
      if(c.corner==="bl"||c.corner==="br")sy=gl-cl;
      for(let dy=0;dy<cl&&sy+dy<gl;dy++)for(let dx=0;dx<cw&&sx+dx<gw;dx++)
        if(sy+dy>=0&&sx+dx>=0)mask[sy+dy][sx+dx]=0;
    }
  }
  return{mask,gw,gl};
}

function rotateMask90(m){
  const rows=m.length,cols=m[0].length;
  const r=Array.from({length:cols},()=>new Uint8Array(rows));
  for(let y=0;y<rows;y++)for(let x=0;x<cols;x++)r[x][rows-1-y]=m[y][x];
  return r;
}

function masksEqual(a,b){
  if(a.length!==b.length||a[0].length!==b[0].length)return false;
  for(let y=0;y<a.length;y++)for(let x=0;x<a[0].length;x++)if(a[y][x]!==b[y][x])return false;
  return true;
}

function getRotations(item){
  const base=buildFootprint(item);
  if(base.gw<=0)return[];
  const rots=[{mask:base.mask,gw:base.gw,gl:base.gl,rw:item.w,rl:item.l}];
  let cur=base.mask;
  // Generate up to 3 more rotations, keep unique
  for(let i=0;i<3;i++){
    cur=rotateMask90(cur);
    const gl2=cur.length,gw2=cur[0].length;
    // Swap real dims for rotated versions
    const rw2=i%2===0?item.l:item.w, rl2=i%2===0?item.w:item.l;
    if(!rots.some(r=>masksEqual(r.mask,cur)))
      rots.push({mask:cur,gw:gw2,gl:gl2,rw:rw2,rl:rl2});
  }
  return rots;
}

// Mask cells → SVG rects (row-span merged for efficiency)
function maskToRects(mask,x0,y0,cw,ch){
  const rects=[];
  for(let dy=0;dy<mask.length;dy++){let sx=null;
    for(let dx=0;dx<=mask[0].length;dx++){
      if(dx<mask[0].length&&mask[dy][dx]){if(sx===null)sx=dx;}
      else if(sx!==null){rects.push({x:x0+sx*cw,y:y0+dy*ch,w:(dx-sx)*cw,h:ch});sx=null;}
    }}
  return rects;
}

// Count filled cells in mask
function maskCount(m){let c=0;for(const r of m)for(const v of r)c+=v;return c;}

// ╔══════════════════════════════════════════════════════════════════╗
// ║  3D TRAILER VOLUME FROM MEASUREMENTS                           ║
// ╚══════════════════════════════════════════════════════════════════╝

function buildMaps(tr){
  const gW=Math.ceil(tr.w/GRID),gL=Math.ceil(tr.l/GRID);
  const ceilMap=Array.from({length:gL},()=>{const r=new Float32Array(gW);r.fill(tr.h);return r;});
  const floorMap=Array.from({length:gL},()=>new Float32Array(gW));
  const validMap=Array.from({length:gL},()=>{const r=new Uint8Array(gW);r.fill(1);return r;});
  const cs=(tr.cs&&tr.cs.length>=2)?[...tr.cs].sort((a,b)=>a.h-b.h):[{h:0,w:tr.w},{h:tr.h,w:tr.w}];
  const lp=(tr.lp&&tr.lp.length>=1)?[...tr.lp].sort((a,b)=>a.d-b.d):[{d:0,w:tr.w},{d:tr.l,w:tr.w}];
  const floorW=cs[0].w;
  for(let gy=0;gy<gL;gy++){const yIn=(gy+.5)*GRID;const wAtY=interp(lp,yIn,"d","w");const wSc=floorW>0?Math.min(wAtY/floorW,1):1;
    for(let gx=0;gx<gW;gx++){const xIn=(gx+.5)*GRID;const xFC=Math.abs(xIn-tr.w/2);const sHW=cs[0].w*wSc/2;
      if(xFC>sHW){validMap[gy][gx]=0;continue;}
      let ceil=cs[cs.length-1].h;let found=false;
      for(let i=cs.length-1;i>=0;i--){const sw=cs[i].w*wSc/2;
        if(sw>=xFC){if(i<cs.length-1){const swA=cs[i+1].w*wSc/2;if(swA<xFC){const t2=(sw-xFC)/(sw-swA);ceil=cs[i].h+t2*(cs[i+1].h-cs[i].h);}}found=true;break;}}
      if(!found)ceil=0;ceilMap[gy][gx]=ceil;}}
  for(const obs of(tr.obs||[])){const sy=Math.floor(obs.y/GRID),sx=Math.floor(obs.x/GRID);
    const ey=Math.min(gL,Math.ceil((obs.y+obs.ol)/GRID)),ex=Math.min(gW,Math.ceil((obs.x+obs.ow)/GRID));
    for(let gy2=sy;gy2<ey;gy2++)for(let gx2=sx;gx2<ex;gx2++)if(gy2>=0&&gx2>=0)floorMap[gy2][gx2]=Math.max(floorMap[gy2][gx2],obs.oh);}
  return{ceilMap,floorMap,validMap,gW,gL};
}

function outlinePath(tr,sc,pad){
  const cs=(tr.cs&&tr.cs.length>=2)?[...tr.cs].sort((a,b)=>a.h-b.h):[{h:0,w:tr.w},{h:tr.h,w:tr.w}];
  const lp=(tr.lp&&tr.lp.length>=1)?[...tr.lp].sort((a,b)=>a.d-b.d):[{d:0,w:tr.w},{d:tr.l,w:tr.w}];
  const fW=cs[0].w;const pts=[];
  for(let i=0;i<=60;i++){const y=(i/60)*tr.l;const wAtY=interp(lp,y,"d","w");const sw=fW>0?Math.min(wAtY/fW,1)*fW:0;const cx=tr.w/2;
    pts.push({lx:pad+(cx-sw/2)*sc,rx:pad+(cx+sw/2)*sc,y:pad+y*sc});}
  let p=`M ${pts[0].lx},${pts[0].y}`;for(const pt of pts)p+=` L ${pt.lx},${pt.y}`;
  p+=` L ${pts[pts.length-1].rx},${pts[pts.length-1].y}`;for(let i=pts.length-1;i>=0;i--)p+=` L ${pts[i].rx},${pts[i].y}`;return p+" Z";
}

// ╔══════════════════════════════════════════════════════════════════╗
// ║  5-PHASE PLACEMENT ENGINE — Mask-aware                         ║
// ╚══════════════════════════════════════════════════════════════════╝

async function computePlan(trailer,items){
  if(!items.length)return{placements:[],loadingOrder:[],unplaced:[],doorFail:[],safety:null,heatmap:null,heatMax:0};
  const maps=buildMaps(trailer);const{ceilMap,floorMap,validMap,gW,gL}=maps;
  const hmap=Array.from({length:gL},(_,gy)=>{const r=new Float32Array(gW);for(let gx=0;gx<gW;gx++)r[gx]=floorMap[gy][gx];return r;});
  const noStack=Array.from({length:gL},()=>new Uint8Array(gW));
  const weightHeat=Array.from({length:gL},()=>new Float32Array(gW));
  let totalPW=0,validC=0;
  for(let gy=0;gy<gL;gy++)for(let gx=0;gx<gW;gx++)if(validMap[gy][gx])validC++;

  const phased=[1,2,3,4,5].map(ph=>items.filter(it=>CAT[it.category].phase===ph).sort((a,b)=>{const wp=WT[b.weight].pri-WT[a.weight].pri;if(wp)return wp;return(b.l*b.w)-(a.l*a.w);}));
  const placements=[];let lW=0,rW=0,fW=0,bW=0;

  for(let pi=0;pi<5;pi++){const phase=pi+1;for(let ii=0;ii<phased[pi].length;ii++){const item=phased[pi][ii];
    if(ii>0&&ii%5===0)await new Promise(r=>setTimeout(r,0));
    const rotations=getRotations(item);
    const res=findPos(hmap,noStack,trailer,item,phase,gW,gL,lW,rW,maps,weightHeat,totalPW,validC,rotations);
    if(res.pos){const p=res.pos,cat=CAT[item.category];
      const mc=maskCount(p.mask);const itemWt=itemWeight(item);const wtPC=mc>0?itemWt/mc:0;
      for(let dy=0;dy<p.gl;dy++)for(let dx=0;dx<p.gw;dx++)if(p.mask[dy][dx]){
        hmap[p.gy+dy][p.gx+dx]=p.z+item.h;
        noStack[p.gy+dy][p.gx+dx]=cat.canBearWeight?0:1;
        weightHeat[p.gy+dy][p.gx+dx]+=wtPC;}
      totalPW+=itemWt;
      const cx2=(p.x+p.rw/2)/trailer.w,cy2=(p.y+p.rl/2)/trailer.l;
      if(cx2<.5)lW+=itemWt;else rW+=itemWt;if(cy2<.55)fW+=itemWt;else bW+=itemWt;}
    placements.push({item,position:res.pos,fits:!!res.pos,phase});}}

  const{grid:heatGrid,max:heatMax}=buildHeatmap(trailer,placements);
  const tot=lW+rW,frontPct=tot>0?fW/tot*100:0,lrImb=tot>0?Math.abs(lW-rW)/tot*100:0;
  const usedVol=placements.reduce((s,p)=>p.position?s+p.item.l*p.item.w*p.item.h:s,0);
  let totVol=0;for(let gy=0;gy<gL;gy++)for(let gx=0;gx<gW;gx++)if(validMap[gy][gx])totVol+=(ceilMap[gy][gx]-floorMap[gy][gx])*GRID*GRID;
  let cogZ=0,cogW2=0;for(const p of placements){if(!p.position)continue;const w=itemWeight(p.item);cogZ+=w*(p.position.z+p.item.h/2);cogW2+=w;}
  const cogHeight=cogW2>0?cogZ/cogW2:0;
  const loadOrder=placements.filter(p=>p.position).sort((a,b)=>{if(a.position.y!==b.position.y)return a.position.y-b.position.y;return a.position.z-b.position.z;}).map((p,i)=>({...p,loadNum:i+1}));
  const dW=trailer.doorW||trailer.w,dH=trailer.doorH||trailer.h;
  const doorFail=placements.filter(p=>p.position).filter(p=>{const d=[p.item.l,p.item.w,p.item.h].sort((a,b)=>a-b);return d[0]>dW||d[0]>dH||d[1]>Math.max(dW,dH);});

  return{placements,loadingOrder:loadOrder,unplaced:placements.filter(p=>!p.position),doorFail,
    safety:{totalWeight:tot,frontPct,backPct:100-frontPct,leftWeight:lW,rightWeight:rW,lrImb,
      overweight:trailer.weightLimit>0&&tot>trailer.weightLimit,volPct:totVol>0?usedVol/totVol*100:0,cogHeight,cogPct:trailer.h>0?cogHeight/trailer.h*100:0,
      // Identity F — Loading Coherence via tightness
      // t = 100/(100+|ε|), ∂I at t = K = 2/3
      tFront: 100/(100+Math.abs(frontPct-57.5)/7.5*50),  // ideal 57.5%, ∂I at 50%/65%
      tLR: 100/(100+lrImb/15*50),                         // ideal 0%, ∂I at 15%
      tCOG: 100/(100+(trailer.h>0?cogHeight/trailer.h*100:0)/50*50), // ∂I at 50% height
      get coherence(){ return Math.min(this.tFront, this.tLR, this.tCOG); },
      get isCoherent(){ return this.coherence >= 2/3; }, // above K = safe
    },
    heatmap:heatGrid,heatMax};
}

function findPos(hmap,noStack,trailer,item,phase,gW,gL,lW,rW,maps,weightHeat,totalPW,validC,rotations){
  const{ceilMap,floorMap,validMap}=maps;const cat=CAT[item.category];
  const avgHeat=validC>0?totalPW/validC:0;
  let best=null,bestS=0; // MAXIMIZE coherence

  for(const rot of rotations){
    if(rot.gw>gW||rot.gl>gL)continue;
    for(let gy=0;gy<=gL-rot.gl;gy++){for(let gx=0;gx<=gW-rot.gw;gx++){
      let maxH=0,blocked=false,invalid=false,minCeil=Infinity,mc=0;
      for(let dy=0;dy<rot.gl&&!invalid&&!blocked;dy++)for(let dx=0;dx<rot.gw&&!invalid&&!blocked;dx++){
        if(!rot.mask[dy][dx])continue;
        mc++;
        if(!validMap[gy+dy][gx+dx]){invalid=true;continue;}
        const ch=hmap[gy+dy][gx+dx];if(ch>maxH)maxH=ch;
        if(ch>floorMap[gy+dy][gx+dx]&&noStack[gy+dy][gx+dx])blocked=true;
        const cl=ceilMap[gy+dy][gx+dx];if(cl<minCeil)minCeil=cl;}
      if(invalid||(blocked&&maxH>0))continue;
      if(cat.floorOnly&&maxH>0)continue;
      if(mc===0||maxH+item.h>minCeil)continue;

      // ═══ SUPPORT FRACTION — structural stability (K=⅔) ═══
      // For stacked placements: cell is "supported" if hmap ≥ maxH−GRID.
      // Below K=⅔ supported footprint → item tips → incoherent → reject.
      if(maxH>0){let sup=0;for(let sy=0;sy<rot.gl;sy++)for(let sx=0;sx<rot.gw;sx++){if(rot.mask[sy][sx]&&hmap[gy+sy][gx+sx]>=maxH-GRID)sup++;}if(mc>0&&sup/mc<2/3)continue;}

      // ═══ TIGHTNESS COHERENCE (Identity F) ═══════════════════
      // t(ε) = 100/(100+|ε|). ∂I at |ε|=50 → t=K=2/3.
      // Each dimension: physical ideal + physical ∂I → ε → t.
      // Coherence = Π(tᵢ). No tuned coefficients.
      const cy=(gy+rot.gl/2)/gL,cx=(gx+rot.gw/2)/gW,iw=itemWeight(item);
      const pri=WT[item.weight].pri;

      // 1. Zone — Eq30 sigmoid → ideal cy; ∂I at ±0.4 trailer lengths
      const sigF=1/(1+Math.exp(-(pri-2.5)*3));
      const cyI=0.75-0.6*sigF;
      const tZone=100/(100+Math.abs(cy-cyI)/0.4*50);

      // 2. Height — floor preferred; ∂I at z=50% trailer height
      const tH=100/(100+(trailer.h>0?(maxH/(trailer.h*0.5))*50:0));

      // 3. L/R balance — ideal 0%; ∂I at 15% imbalance
      const nL=cx<.5?lW+iw:lW,nR=cx>=.5?rW+iw:rW,totLR=nL+nR;
      const tBal=100/(100+(totLR>0?Math.abs(nL-nR)/totLR*100/15*50:0));

      // 4. Heat — ideal ≤ avg; ∂I at 3× avg; front tolerance via sigmoid
      let lH=0;for(let dy=0;dy<rot.gl;dy++)for(let dx=0;dx<rot.gw;dx++)if(rot.mask[dy][dx])lH+=weightHeat[gy+dy][gx+dx];
      lH/=mc||1;
      const hR=avgHeat>0?Math.max(0,lH/avgHeat-1):0;
      const tHeat=100/(100+(hR/2*50)*(1-sigF*0.6));

      // 5. Compactness — wall/neighbor adj; ∂I when isolated
      let adj=0;if(gx===0)adj+=3;if(gx+rot.gw>=gW)adj+=3;if(gy===0)adj+=4;if(gy+rot.gl>=gL)adj+=1;
      if(gx>0){for(let dy=0;dy<rot.gl;dy++)if(rot.mask[dy][0]&&hmap[gy+dy][gx-1]>floorMap[gy+dy][gx-1]){adj+=2;break;}}
      if(gx+rot.gw<gW){for(let dy=0;dy<rot.gl;dy++)if(rot.mask[dy][rot.gw-1]&&hmap[gy+dy][gx+rot.gw]>floorMap[gy+dy][gx+rot.gw]){adj+=2;break;}}
      if(gy>0){for(let dx=0;dx<rot.gw;dx++)if(rot.mask[0][dx]&&hmap[gy-1][gx+dx]>floorMap[gy-1][gx+dx]){adj+=2;break;}}
      if(gy+rot.gl<gL){for(let dx=0;dx<rot.gw;dx++)if(rot.mask[rot.gl-1][dx]&&hmap[gy+rot.gl][gx+dx]>floorMap[gy+rot.gl][gx+dx]){adj+=1;break;}}
      const tCompact=100/(100+Math.max(0,(10-adj)/10)*50);

      // Phase modifier (structural role enhancement)
      let pM=1.0;
      if(phase===3)pM=tCompact>0.75?1.2:0.8;
      if(phase===5)pM=tCompact>0.7?1.3:0.7;
      if(phase===4&&(gx===0||gx+rot.gw>=gW))pM=1.15;

      const coherence=tZone*tH*tBal*tHeat*tCompact*pM;
      if(coherence>bestS){bestS=coherence;best={gx,gy,gw:rot.gw,gl:rot.gl,x:gx*GRID,y:gy*GRID,z:maxH,rw:rot.rw,rl:rot.rl,mask:rot.mask};}
    }}}
  return{pos:best};
}

function buildHeatmap(trailer,placements){
  const cols=Math.ceil(trailer.w/HEAT),rows=Math.ceil(trailer.l/HEAT);
  const grid=Array.from({length:rows},()=>new Float32Array(cols));let max=0;
  for(const p of placements){if(!p.position)continue;const pos=p.position,w=itemWeight(p.item);
    const mc=pos.mask?maskCount(pos.mask):pos.gw*pos.gl;if(mc<=0)continue;
    const wpc=w/mc*(GRID*GRID)/(HEAT*HEAT);
    for(let dy=0;dy<pos.gl;dy++)for(let dx=0;dx<pos.gw;dx++){
      if(pos.mask&&!pos.mask[dy][dx])continue;
      const r=Math.floor((pos.y+dy*GRID)/HEAT),c=Math.floor((pos.x+dx*GRID)/HEAT);
      if(r>=0&&r<rows&&c>=0&&c<cols){grid[r][c]+=wpc;if(grid[r][c]>max)max=grid[r][c];}}}
  return{grid,max};
}

function zoneLabel(pos,trailer){const cy=(pos.y+pos.rl/2)/trailer.l,cx=(pos.x+pos.rw/2)/trailer.w;return`${cy<.35?"Front":cy<.65?"Mid":"Back"}-${cx<.35?"Left":cx>.65?"Right":"Center"}${pos.z>0?`, stacked ${fmtD(pos.z)} up`:", floor"}`;}

// ╔══════════════════════════════════════════════════════════════════╗
// ║  COMPONENT                                                       ║
// ╚══════════════════════════════════════════════════════════════════╝

export default function TrailerLoadPlanner(){
  const[step,setStep]=useState("trailer");
  const[tFt,setTFt]=useState({l:"",w:"",h:""});const[tIn,setTIn]=useState({l:"",w:"",h:""});const[wLim,setWLim]=useState("");
  const[trailer,setTrailer]=useState(null);
  const[csRows,setCsRows]=useState([]);const[csTH,setCsTH]=useState("");const[csTW,setCsTW]=useState("");
  const[lpRows,setLpRows]=useState([]);const[lpTD,setLpTD]=useState("");const[lpTW,setLpTW]=useState("");
  const[obsRows,setObsRows]=useState([]);const[obsTN,setObsTN]=useState("");const[obsTY,setObsTY]=useState("");const[obsTL,setObsTL]=useState("");const[obsTW,setObsTW]=useState("");const[obsTH,setObsTH]=useState("");const[obsTSide,setObsTSide]=useState("left");
  const[doorWIn,setDoorWIn]=useState("");const[doorHIn,setDoorHIn]=useState("");

  const[items,setItems]=useState([]);
  const[cn,setCn]=useState("");const[cL,setCL]=useState("");const[cW,setCW]=useState("");const[cH,setCH]=useState("");
  const[cWt,setCWt]=useState("");const[cCat,setCCat]=useState("");
  const[cSurf,setCSurf]=useState("flat");
  const[cCuts,setCCuts]=useState([]); // [{id,corner,cw,cl}]
  const[cutCorner,setCutCorner]=useState("br");const[cutW,setCutW]=useState("");const[cutL,setCutL]=useState("");
  const[editId,setEditId]=useState(null);
  const[plan,setPlan]=useState(null);const[selItem,setSelItem]=useState(null);const[viewMode,setViewMode]=useState("tetris");
  const[layerStep,setLayerStep]=useState(999);const[wtVis,setWtVis]=useState({extra:true,heavy:true,medium:true,light:true});
  const[cExact,setCExact]=useState("");
  const[undoStack,setUndoStack]=useState([]);
  const[err,setErr]=useState("");
  const[loading,setLoading]=useState(true);
  const[savedMsg,setSavedMsg]=useState("");
  const[showAdv,setShowAdv]=useState(false);
  const[computing,setComputing]=useState(false);
  const[storageOk,setStorageOk]=useState(true);

  // ═══ PERSISTENCE: Load on mount — manual save has priority ═══
  useEffect(()=>{
    (async()=>{
      if(!window.storage||typeof window.storage.get!=='function'){setStorageOk(false);setLoading(false);return;}
      try{
        let src=null;
        try{await window.storage.get('tlp-manual');src='manual';}catch(e){}
        if(!src){try{await window.storage.get('tlp-auto');src='auto';}catch(e){}}
        if(src){
          try{const res=await window.storage.get(`tlp-${src}`);if(res&&res.value){const d=JSON.parse(res.value);
            if(d.trailer)setTrailer(d.trailer);
            if(d.items){setItems(d.items);const mx=d.items.reduce((m,x)=>Math.max(m,x.id||0),0);_id=mx+1;}
            if(d.setup){const s=d.setup;
              if(s.tFt)setTFt(s.tFt);if(s.tIn)setTIn(s.tIn);if(s.wLim!==undefined)setWLim(s.wLim);
              if(s.csRows)setCsRows(s.csRows);if(s.lpRows)setLpRows(s.lpRows);if(s.obsRows)setObsRows(s.obsRows);
              if(s.doorWIn!==undefined)setDoorWIn(s.doorWIn);if(s.doorHIn!==undefined)setDoorHIn(s.doorHIn);
              if(s.step)setStep(s.step);}
          }}catch(e){}
        }
      }catch(e){console.error('TLP load:',e);}
      setLoading(false);
    })();
  },[]);

  // ═══ PERSISTENCE: Auto-save all state to single key ═══
  useEffect(()=>{if(loading||!storageOk)return;(async()=>{try{await window.storage.set('tlp-auto',JSON.stringify({trailer,items,setup:{tFt,tIn,wLim,csRows,lpRows,obsRows,doorWIn,doorHIn,step}}));}catch(e){}})();},[trailer,items,tFt,tIn,wLim,csRows,lpRows,obsRows,doorWIn,doorHIn,step,loading,storageOk]);

  // ═══ PERSISTENCE: Manual save handler ═══
  const manualSave=useCallback(async()=>{
    try{
      await window.storage.set('tlp-manual',JSON.stringify({trailer,items,setup:{tFt,tIn,wLim,csRows,lpRows,obsRows,doorWIn,doorHIn,step}}));
      setSavedMsg("Saved ✓");setTimeout(()=>setSavedMsg(""),2000);
    }catch(e){console.error('TLP save:',e);setSavedMsg("Save failed ✗");setTimeout(()=>setSavedMsg(""),3000);}
  },[trailer,items,tFt,tIn,wLim,csRows,lpRows,obsRows,doorWIn,doorHIn,step]);

  const setTrailerDims=()=>{const l=toIn(tFt.l,tIn.l),w=toIn(tFt.w,tIn.w),h=toIn(tFt.h,tIn.h),wl=parseFloat(wLim)||0;
    if(l<=0||w<=0||h<=0){setErr("All dimensions must be positive.");return;}
    if(l>600||w>120||h>120){setErr("Dimensions seem too large — check your numbers.");return;}
    setErr("");
    const cs=csRows.length>=2?csRows.map(r=>({h:r.h,w:r.w})):[{h:0,w},{h,w}];
    const lp=lpRows.length>=1?lpRows.map(r=>({d:r.d,w:r.w})):[{d:0,w},{d:l,w}];
    const obs=obsRows.map(r=>{const ox=r.side==="left"?0:r.side==="right"?w-r.ow:(w-r.ow)/2;return{x:ox,y:r.y,ol:r.ol,ow:r.ow,oh:r.oh};});
    setTrailer({l,w,h,weightLimit:wl,cs,lp,obs,doorW:parseFloat(doorWIn)||w,doorH:parseFloat(doorHIn)||h});setStep("items");};

  const addItem=()=>{const l=parseFloat(cL)||0,w=parseFloat(cW)||0,h=parseFloat(cH)||0;const ev=parseFloat(cExact);
    if(!cn.trim()){setErr("Enter a name.");return;}
    if(l<=0||w<=0||h<=0){setErr("All dimensions must be positive.");return;}
    if(!cCat){setErr("Select a category.");return;}
    if(!cWt){setErr("Select a weight feel.");return;}
    if(!isNaN(ev)&&ev<0){setErr("Weight cannot be negative.");return;}
    if(trailer&&(l>trailer.l||w>trailer.w||h>trailer.h)){setErr("⚠ Item is larger than trailer in one or more dimensions.");setTimeout(()=>setErr(p=>p.startsWith("⚠")?"":p),3000);}else{setErr("");}
    const newIt={id:editId!==null?editId:nid(),name:cn.trim(),l,w,h,weight:cWt,exactLbs:isNaN(ev)?null:ev,category:cCat,surface:cSurf,cutouts:[...cCuts]};
    if(editId!==null){setItems(p=>p.map(it=>it.id===editId?newIt:it));setEditId(null);}
    else setItems(p=>[...p,newIt]);
    setCn("");setCL("");setCW("");setCH("");setCWt("");setCCat("");setCSurf("flat");setCCuts([]);setCExact("");};
  const startEdit=it=>{setEditId(it.id);setCn(it.name);setCL(String(it.l));setCW(String(it.w));setCH(String(it.h));setCWt(it.weight);setCCat(it.category);setCSurf(it.surface||"flat");setCCuts(it.cutouts||[]);setCExact(it.exactLbs!=null?String(it.exactLbs):"");};
  const cancelEdit=()=>{setEditId(null);setCn("");setCL("");setCW("");setCH("");setCWt("");setCCat("");setCSurf("flat");setCCuts([]);setCExact("");};
  const deleteItem=id=>{const victim=items.find(it=>it.id===id);if(victim)setUndoStack(p=>[...p,victim]);setItems(p=>p.filter(it=>it.id!==id));if(editId===id)cancelEdit();};
  const addCut=()=>{const cw2=parseFloat(cutW)||0,cl2=parseFloat(cutL)||0;if(cw2>0&&cl2>0){setCCuts(p=>[...p,{id:nid(),corner:cutCorner,cw:cw2,cl:cl2}]);setCutW("");setCutL("");}};
  const undoDelete=()=>{if(undoStack.length===0)return;const last=undoStack[undoStack.length-1];setUndoStack(p=>p.slice(0,-1));setItems(p=>[...p,last]);};
  const calc=async()=>{if(!trailer||!items.length||computing)return;setComputing(true);try{const result=await computePlan(trailer,items);setPlan(result);setStep("plan");setLayerStep(999);setSelItem(null);}finally{setComputing(false);}};
  const runW=useMemo(()=>items.reduce((s,it)=>s+itemWeight(it),0),[items]);

  const inp={width:"100%",padding:"12px",fontSize:"17px",border:"2px solid #d1d5db",borderRadius:"10px",boxSizing:"border-box",backgroundColor:"#fff"};
  const inpS={...inp,fontSize:"15px",padding:"10px"};
  const btnP={width:"100%",padding:"16px",fontSize:"18px",fontWeight:"700",color:"#fff",backgroundColor:"#2563eb",border:"none",borderRadius:"12px",cursor:"pointer"};
  const btnS={...btnP,backgroundColor:"#6b7280",fontSize:"15px",padding:"12px"};
  const btnSm={padding:"8px 14px",fontSize:"14px",fontWeight:"700",color:"#fff",backgroundColor:"#2563eb",border:"none",borderRadius:"8px",cursor:"pointer"};
  const card={backgroundColor:"#fff",borderRadius:"14px",padding:"18px",marginBottom:"14px",boxShadow:"0 1px 4px rgba(0,0,0,0.08)",border:"1px solid #e5e7eb"};
  const secH={fontSize:"15px",fontWeight:"700",color:"#1f2937",marginBottom:"6px"};
  const hint={fontSize:"11px",color:"#9ca3af"};

  // ═══════════════ TRAILER SETUP ═══════════════
  if(step==="trailer"){return(
    <div style={{padding:"16px",maxWidth:480,margin:"0 auto",fontFamily:"system-ui,-apple-system,sans-serif"}}>
      <h1 style={{fontSize:"22px",fontWeight:"800",marginBottom:"2px",color:"#111"}}>🚛 Trailer Load Planner</h1>
      <p style={{color:"#6b7280",fontSize:"13px",marginBottom:"20px"}}>Measure it. Enter the numbers. True 3D model.</p>
      {err&&<div style={{padding:"12px",backgroundColor:"#fef2f2",border:"1px solid #fca5a5",borderRadius:"10px",marginBottom:"14px"}}><p style={{fontSize:"13px",color:"#991b1b",fontWeight:"600",margin:0}}>{err}</p></div>}
      <div style={card}>
        <h2 style={{fontSize:"17px",fontWeight:"700",marginBottom:"16px",color:"#1f2937"}}>Bounding Dimensions</h2>
        {["l","w","h"].map(d=>{const lab=d==="l"?"Total Length":d==="w"?"Maximum Width":"Maximum Height";
          return(<div key={d} style={{marginBottom:"14px"}}><label style={{fontSize:"15px",fontWeight:"600",color:"#374151",display:"block",marginBottom:"6px"}}>{lab}</label>
            <div style={{display:"flex",gap:"8px"}}><input type="number" inputMode="numeric" placeholder="ft" value={tFt[d]} onChange={e=>setTFt(p=>({...p,[d]:e.target.value}))} style={inp}/>
              <input type="number" inputMode="numeric" placeholder="in" value={tIn[d]} onChange={e=>setTIn(p=>({...p,[d]:e.target.value}))} style={inp}/></div></div>);})}
        <label style={{fontSize:"15px",fontWeight:"600",color:"#374151",display:"block",marginBottom:"6px"}}>Weight Limit (lbs)</label>
        <input type="number" inputMode="numeric" placeholder="optional" value={wLim} onChange={e=>setWLim(e.target.value)} style={inp}/>
      </div>
      <button onClick={()=>setShowAdv(p=>!p)} style={{width:"100%",padding:"12px",fontSize:"15px",fontWeight:"700",color:"#2563eb",backgroundColor:"transparent",border:"2px solid #2563eb",borderRadius:"10px",cursor:"pointer",marginBottom:"14px"}}>{showAdv?"▼ Hide Shape Details":"▶ Refine Shape"}</button>
      {showAdv&&<>
      <div style={card}><h2 style={secH}>Cross-Section Profile</h2><p style={{...hint,marginBottom:"10px"}}>Width at different heights. Skip if rectangular.</p>
        {csRows.sort((a,b)=>a.h-b.h).map(r=><div key={r.id} style={{display:"flex",alignItems:"center",gap:"6px",marginBottom:"6px"}}><span style={{fontSize:"14px",color:"#374151",flex:1}}>At {fmtD(r.h)} high → {fmtD(r.w)} wide</span>
          <button onClick={()=>setCsRows(p=>p.filter(x=>x.id!==r.id))} style={{color:"#dc2626",background:"none",border:"none",fontSize:"18px",cursor:"pointer"}}>✕</button></div>)}
        <div style={{display:"flex",gap:"6px",alignItems:"end"}}><div style={{flex:1}}><label style={hint}>Height"</label><input type="number" inputMode="decimal" value={csTH} onChange={e=>setCsTH(e.target.value)} style={inpS}/></div>
          <div style={{flex:1}}><label style={hint}>Width"</label><input type="number" inputMode="decimal" value={csTW} onChange={e=>setCsTW(e.target.value)} style={inpS}/></div>
          <button onClick={()=>{const h2=parseFloat(csTH),w2=parseFloat(csTW);if(h2>=0&&w2>0){setCsRows(p=>[...p,{id:nid(),h:h2,w:w2}]);setCsTH("");setCsTW("");}}} style={btnSm}>+</button></div></div>
      <div style={card}><h2 style={secH}>Width Along Length</h2><p style={{...hint,marginBottom:"10px"}}>Width at different distances from front. Skip if constant.</p>
        {lpRows.sort((a,b)=>a.d-b.d).map(r=><div key={r.id} style={{display:"flex",alignItems:"center",gap:"6px",marginBottom:"6px"}}><span style={{fontSize:"14px",color:"#374151",flex:1}}>At {fmtD(r.d)} back → {fmtD(r.w)} wide</span>
          <button onClick={()=>setLpRows(p=>p.filter(x=>x.id!==r.id))} style={{color:"#dc2626",background:"none",border:"none",fontSize:"18px",cursor:"pointer"}}>✕</button></div>)}
        <div style={{display:"flex",gap:"6px",alignItems:"end"}}><div style={{flex:1}}><label style={hint}>From front"</label><input type="number" inputMode="decimal" value={lpTD} onChange={e=>setLpTD(e.target.value)} style={inpS}/></div>
          <div style={{flex:1}}><label style={hint}>Width"</label><input type="number" inputMode="decimal" value={lpTW} onChange={e=>setLpTW(e.target.value)} style={inpS}/></div>
          <button onClick={()=>{const d2=parseFloat(lpTD),w2=parseFloat(lpTW);if(d2>=0&&w2>0){setLpRows(p=>[...p,{id:nid(),d:d2,w:w2}]);setLpTD("");setLpTW("");}}} style={btnSm}>+</button></div></div>
      <div style={card}><h2 style={secH}>Obstacles</h2><p style={{...hint,marginBottom:"10px"}}>Wheel wells, built-ins.</p>
        {obsRows.map(r=><div key={r.id} style={{display:"flex",alignItems:"center",gap:"6px",marginBottom:"6px"}}><span style={{fontSize:"13px",color:"#374151",flex:1}}>{r.name} · {r.side}</span>
          <button onClick={()=>setObsRows(p=>p.filter(x=>x.id!==r.id))} style={{color:"#dc2626",background:"none",border:"none",fontSize:"18px",cursor:"pointer"}}>✕</button></div>)}
        <input type="text" placeholder="Name" value={obsTN} onChange={e=>setObsTN(e.target.value)} style={{...inpS,marginBottom:"6px"}}/>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"6px",marginBottom:"6px"}}>
          {[["From front\"",obsTY,setObsTY],["Length\"",obsTL,setObsTL],["Width\"",obsTW,setObsTW],["Height\"",obsTH,setObsTH]].map(([l,v,s],i)=>
            <div key={i}><label style={hint}>{l}</label><input type="number" inputMode="decimal" value={v} onChange={e=>s(e.target.value)} style={inpS}/></div>)}</div>
        <div style={{display:"flex",gap:"6px",alignItems:"end"}}><div style={{flex:1}}><div style={{display:"flex",gap:"4px"}}>{["left","right","center"].map(s2=>
          <button key={s2} onClick={()=>setObsTSide(s2)} style={{...inpS,flex:1,textAlign:"center",fontWeight:"600",backgroundColor:obsTSide===s2?"#2563eb":"#fff",color:obsTSide===s2?"#fff":"#374151",cursor:"pointer"}}>{s2}</button>)}</div></div>
          <button onClick={()=>{const y=parseFloat(obsTY),ol=parseFloat(obsTL),ow=parseFloat(obsTW),oh=parseFloat(obsTH);if(obsTN.trim()&&y>=0&&ol>0&&ow>0&&oh>0){setObsRows(p=>[...p,{id:nid(),name:obsTN.trim(),y,ol,ow,oh,side:obsTSide}]);setObsTN("");setObsTY("");setObsTL("");setObsTW("");setObsTH("");}}} style={btnSm}>+</button></div></div>
      <div style={card}><h2 style={secH}>Door Opening</h2><div style={{display:"flex",gap:"8px"}}><div style={{flex:1}}><label style={hint}>Width" (blank=full)</label><input type="number" inputMode="decimal" value={doorWIn} onChange={e=>setDoorWIn(e.target.value)} style={inpS}/></div>
          <div style={{flex:1}}><label style={hint}>Height" (blank=full)</label><input type="number" inputMode="decimal" value={doorHIn} onChange={e=>setDoorHIn(e.target.value)} style={inpS}/></div></div></div>
      </>}
      <button onClick={setTrailerDims} style={btnP}>Set Trailer →</button></div>);}

  // ═══════════════ ADD ITEMS ═══════════════
  if(step==="items"){const valid=cn.trim()&&parseFloat(cL)>0&&parseFloat(cW)>0&&parseFloat(cH)>0&&cWt&&cCat;
    return(<div style={{padding:"16px",maxWidth:480,margin:"0 auto",fontFamily:"system-ui,-apple-system,sans-serif"}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"4px"}}><h1 style={{fontSize:"20px",fontWeight:"800",color:"#111",margin:0}}>Add Items</h1>
        <button onClick={()=>{setErr("");setStep("trailer");}} style={{fontSize:"14px",color:"#2563eb",background:"none",border:"none",cursor:"pointer",fontWeight:"600"}}>← Trailer</button></div>
      {trailer&&<p style={{color:"#6b7280",fontSize:"12px",marginBottom:"14px"}}>Trailer: {fmtD3(trailer.l,trailer.w,trailer.h)}{trailer.weightLimit>0?` · ${trailer.weightLimit} lbs`:""}</p>}
      {err&&<div style={{padding:"12px",backgroundColor:err.startsWith("⚠")?"#fffbeb":"#fef2f2",border:`1px solid ${err.startsWith("⚠")?"#fbbf24":"#fca5a5"}`,borderRadius:"10px",marginBottom:"14px"}}><p style={{fontSize:"13px",color:err.startsWith("⚠")?"#92400e":"#991b1b",fontWeight:"600",margin:0}}>{err}</p></div>}
      {items.length>0&&<div style={{...card,padding:"12px 16px",display:"flex",justifyContent:"space-between",alignItems:"center"}}><span style={{fontWeight:"700",fontSize:"15px",color:"#1f2937"}}>~{runW} lbs</span><span style={{fontSize:"14px",color:"#6b7280"}}>{items.length} item{items.length!==1?"s":""}</span></div>}
      <div style={card}>
        <h2 style={{fontSize:"16px",fontWeight:"700",marginBottom:"14px",color:"#1f2937"}}>{editId!==null?"Edit":"New"} Item</h2>
        <p style={{fontSize:"14px",fontWeight:"600",color:"#374151",marginBottom:"8px"}}>Category</p>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:"6px",marginBottom:"14px"}}>
          {CAT_KEYS.map(k=>{const c=CAT[k],sel=cCat===k;return<button key={k} onClick={()=>{setCCat(k);if(!cWt)setCWt(c.defaultWt);}}
            style={{padding:"10px 4px",fontSize:"13px",fontWeight:"700",border:sel?`3px solid ${CC[k]}`:"2px solid #d1d5db",borderRadius:"10px",backgroundColor:sel?CC[k]:"#fff",color:sel?"#fff":CC[k],cursor:"pointer",textAlign:"center"}}>
            <span style={{fontSize:"20px",display:"block"}}>{c.emoji}</span>{c.label}</button>;})}</div>
        <input type="text" placeholder="Item name" value={cn} onChange={e=>setCn(e.target.value)} style={{...inp,marginBottom:"12px"}}/>
        <div style={{display:"flex",gap:"8px",marginBottom:"4px"}}>
          <input type="number" inputMode="decimal" placeholder='L"' value={cL} onChange={e=>setCL(e.target.value)} style={{...inp,textAlign:"center"}}/>
          <input type="number" inputMode="decimal" placeholder='W"' value={cW} onChange={e=>setCW(e.target.value)} style={{...inp,textAlign:"center"}}/>
          <input type="number" inputMode="decimal" placeholder='H"' value={cH} onChange={e=>setCH(e.target.value)} style={{...inp,textAlign:"center"}}/></div>
        <p style={{...hint,marginBottom:"12px"}}>L × W × H in inches</p>

        {/* Surface + Cutouts */}
        <p style={{fontSize:"14px",fontWeight:"600",color:"#374151",marginBottom:"6px"}}>Surface</p>
        <div style={{display:"flex",gap:"6px",marginBottom:"12px"}}>
          {[["flat","Flat"],["round","Round/Cyl"]].map(([k,lb])=>
            <button key={k} onClick={()=>setCSurf(k)} style={{flex:1,padding:"10px",fontSize:"14px",fontWeight:"700",border:cSurf===k?"3px solid #2563eb":"2px solid #d1d5db",borderRadius:"10px",backgroundColor:cSurf===k?"#2563eb":"#fff",color:cSurf===k?"#fff":"#374151",cursor:"pointer"}}>{lb}</button>)}</div>

        {cSurf==="flat"&&<>
          <p style={{fontSize:"14px",fontWeight:"600",color:"#374151",marginBottom:"6px"}}>Cutouts <span style={{fontWeight:"400",color:"#9ca3af"}}>(optional — where material is missing)</span></p>
          {cCuts.map(c=><div key={c.id} style={{display:"flex",alignItems:"center",gap:"6px",marginBottom:"4px"}}>
            <span style={{fontSize:"13px",color:"#374151",flex:1}}>{c.corner.toUpperCase()} corner: {fmtD(c.cw)} × {fmtD(c.cl)}</span>
            <button onClick={()=>setCCuts(p=>p.filter(x=>x.id!==c.id))} style={{color:"#dc2626",background:"none",border:"none",fontSize:"16px",cursor:"pointer"}}>✕</button></div>)}
          <div style={{display:"flex",gap:"6px",alignItems:"end",marginBottom:"12px"}}>
            <div><label style={hint}>Corner</label>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"2px",width:"64px"}}>
                {[["fl","◤"],["fr","◥"],["bl","◣"],["br","◢"]].map(([k,sym])=>
                  <button key={k} onClick={()=>setCutCorner(k)} style={{padding:"6px",fontSize:"16px",border:cutCorner===k?"2px solid #2563eb":"1px solid #d1d5db",borderRadius:"4px",backgroundColor:cutCorner===k?"#eff6ff":"#fff",cursor:"pointer"}}>{sym}</button>)}</div></div>
            <div style={{flex:1}}><label style={hint}>Width"</label><input type="number" inputMode="decimal" value={cutW} onChange={e=>setCutW(e.target.value)} style={inpS}/></div>
            <div style={{flex:1}}><label style={hint}>Depth"</label><input type="number" inputMode="decimal" value={cutL} onChange={e=>setCutL(e.target.value)} style={inpS}/></div>
            <button onClick={addCut} style={btnSm}>+</button>
          </div></>}

        <p style={{fontSize:"14px",fontWeight:"600",color:"#374151",marginBottom:"8px"}}>Weight Feel</p>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"8px",marginBottom:"14px"}}>
          {WT_KEYS.map(k=>{const c=WT[k],sel=cWt===k;return<button key={k} onClick={()=>setCWt(k)}
            style={{padding:"12px 6px",fontSize:"15px",fontWeight:"700",border:sel?`3px solid ${c.color}`:"2px solid #d1d5db",borderRadius:"10px",backgroundColor:sel?c.color:"#fff",color:sel?"#fff":c.color,cursor:"pointer"}}>
            {c.label}<br/><span style={{fontSize:"11px",fontWeight:"400",opacity:.85}}>~{c.lbs} lbs</span></button>;})}</div>
        <div style={{marginBottom:"14px"}}><label style={{fontSize:"13px",color:"#6b7280",display:"block",marginBottom:"6px"}}>Or enter exact weight (lbs)</label>
          <input type="number" inputMode="decimal" placeholder="optional — overrides feel" value={cExact} onChange={e=>setCExact(e.target.value)} style={inpS}/></div>
        <button onClick={addItem} disabled={!valid} style={{...btnP,backgroundColor:valid?"#2563eb":"#d1d5db",cursor:valid?"pointer":"default"}}>{editId!==null?"Save":"Add Item"}</button>
        {editId!==null&&<button onClick={cancelEdit} style={{...btnS,marginTop:"8px"}}>Cancel</button>}
      </div>
      {undoStack.length>0&&<button onClick={undoDelete} style={{width:"100%",padding:"12px",fontSize:"14px",fontWeight:"700",color:"#2563eb",backgroundColor:"#eff6ff",border:"2px solid #2563eb",borderRadius:"10px",cursor:"pointer",marginBottom:"14px"}}>↩ Undo delete: {undoStack[undoStack.length-1].name}</button>}
      {items.length>0&&<><div style={{maxHeight:"280px",overflowY:"auto",marginBottom:"14px"}}>
        {items.map(it=>{const wc=WT[it.weight],cc=CAT[it.category];const hasShape=it.surface==="round"||(it.cutouts&&it.cutouts.length>0);
          return<div key={it.id} style={{...card,padding:"10px 12px",borderLeft:`5px solid ${CC[it.category]}`,display:"flex",justifyContent:"space-between",alignItems:"center"}}>
            <div style={{minWidth:0,flex:1}}><div style={{fontWeight:"700",fontSize:"14px",color:"#1f2937",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{cc.emoji} {it.name}{hasShape?" ○":""}</div>
              <div style={{fontSize:"12px",color:"#6b7280"}}>{fmtD3(it.l,it.w,it.h)} · {it.exactLbs?`${it.exactLbs} lbs`:wc.label}{it.cutouts&&it.cutouts.length>0?` · ${it.cutouts.length} cutout${it.cutouts.length>1?"s":""}`:""}{it.surface==="round"?" · round":""}</div></div>
            <div style={{display:"flex",gap:"6px",flexShrink:0}}>
              <button onClick={()=>startEdit(it)} style={{...btnS,padding:"6px 10px",fontSize:"12px",width:"auto"}}>Edit</button>
              <button onClick={()=>deleteItem(it.id)} style={{padding:"6px 10px",fontSize:"12px",color:"#fff",backgroundColor:"#dc2626",border:"none",borderRadius:"8px",cursor:"pointer"}}>✕</button></div></div>;})}</div>
        <button onClick={calc} disabled={computing} style={{...btnP,backgroundColor:computing?"#9ca3af":"#059669",marginBottom:"10px",cursor:computing?"wait":"pointer"}}>{computing?"Computing…":"Calculate Loading Plan"}</button>
        <button onClick={manualSave} disabled={!storageOk} style={{...btnS,marginBottom:"10px",opacity:storageOk?1:0.4}}>{!storageOk?"💾 N/A":savedMsg||"💾 Save"}</button></>}
    </div>);}

  // ═══════════════ PLAN ═══════════════
  if(step==="plan"&&plan&&trailer){
    const{safety,loadingOrder,unplaced,doorFail,heatmap,heatMax}=plan;
    const mxW=360,mxH=480,sc=Math.min(mxW/trailer.w,mxH/trailer.l),svgW=trailer.w*sc,svgH=trailer.l*sc,pad=36;
    const oPath=outlinePath(trailer,sc,pad);
    const fcol=safety.frontPct>=50&&safety.frontPct<=65?"#16a34a":safety.frontPct>65?"#ca8a04":"#dc2626";
    const bcol=safety.lrImb<=10?"#16a34a":safety.lrImb<=20?"#ca8a04":"#dc2626";
    const ccol=safety.cogPct<=50?"#16a34a":safety.cogPct<=65?"#ca8a04":"#dc2626";
    const obsRects=(trailer.obs||[]).map((o,i)=>({key:i,x:pad+o.x*sc,y:pad+o.y*sc,w:o.ow*sc,h:o.ol*sc}));
    const cellW=GRID*sc,cellH=GRID*sc;

    const cohVal=safety.coherence||0; const K=2/3;
    const cohCol=cohVal>=K+0.1?"#16a34a":cohVal>=K?"#ca8a04":"#dc2626";
    const cohLabel=cohVal>=K+0.1?"COHERENT":cohVal>=K?"AT ∂I":"INCOHERENT";

    return(<div style={{padding:"16px",maxWidth:480,margin:"0 auto",fontFamily:"system-ui,-apple-system,sans-serif"}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"12px"}}><h1 style={{fontSize:"20px",fontWeight:"800",color:"#111",margin:0}}>Loading Plan</h1>
        <button onClick={()=>{setErr("");setStep("items");}} style={{fontSize:"14px",color:"#2563eb",background:"none",border:"none",cursor:"pointer",fontWeight:"600"}}>← Items</button></div>

      {/* Loading Coherence — Identity F ∂I boundary */}
      <div style={{...card,borderColor:cohCol,borderWidth:"2px",background:`linear-gradient(135deg, ${cohCol}11, #fff)`}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}>
          <div><div style={{fontSize:"13px",fontWeight:"700",color:"#374151"}}>Loading Coherence</div>
            <div style={{fontSize:"11px",color:"#6b7280"}}>t = min(tᵢ) — Identity F, ∂I at K=⅔</div></div>
          <div style={{textAlign:"right"}}><div style={{fontSize:"28px",fontWeight:"800",color:cohCol}}>{(cohVal*100).toFixed(0)}%</div>
            <div style={{fontSize:"11px",fontWeight:"700",color:cohCol}}>{cohLabel}</div></div></div>
        {/* Tightness bar */}
        <div style={{marginTop:"10px",height:"8px",backgroundColor:"#e5e7eb",borderRadius:"4px",overflow:"hidden",position:"relative"}}>
          <div style={{height:"100%",width:`${Math.min(cohVal*100,100)}%`,backgroundColor:cohCol,borderRadius:"4px",transition:"width 0.3s"}}/>
          <div style={{position:"absolute",left:`${K*100}%`,top:"-2px",width:"2px",height:"12px",backgroundColor:"#1f2937"}}/>
        </div>
        <div style={{display:"flex",justifyContent:"space-between",marginTop:"2px"}}><span style={{fontSize:"9px",color:"#9ca3af"}}>0%</span><span style={{fontSize:"9px",color:"#1f2937",fontWeight:"700"}}>K=⅔</span><span style={{fontSize:"9px",color:"#9ca3af"}}>100%</span></div>
      </div>

      <div style={card}><h2 style={{fontSize:"16px",fontWeight:"700",marginBottom:"12px",color:"#1f2937"}}>Metrics</h2>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:"8px"}}>
          {[{v:`${safety.frontPct.toFixed(0)}%`,l:"Front Wt",c:fcol,n:safety.frontPct>=50&&safety.frontPct<=65?"✓":"✗"},{v:`${safety.lrImb.toFixed(0)}%`,l:"L/R Imbal",c:bcol,n:safety.lrImb<=10?"✓":"⚠"},{v:`${safety.cogPct.toFixed(0)}%`,l:"CoG Ht",c:ccol,n:safety.cogPct<=50?"✓":"⚠"}].map((m,i)=>
            <div key={i} style={{padding:"8px",borderRadius:"10px",backgroundColor:"#f9fafb",textAlign:"center"}}><div style={{fontSize:"22px",fontWeight:"800",color:m.c}}>{m.v}</div><div style={{fontSize:"11px",color:"#6b7280"}}>{m.l}</div><div style={{fontSize:"10px",color:m.c,fontWeight:"600"}}>{m.n}</div></div>)}</div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"8px",marginTop:"8px"}}>
          <div style={{padding:"8px",borderRadius:"10px",backgroundColor:"#f9fafb",textAlign:"center"}}><div style={{fontSize:"18px",fontWeight:"800",color:safety.overweight?"#dc2626":"#1f2937"}}>~{safety.totalWeight} lbs</div><div style={{fontSize:"11px",color:"#6b7280"}}>Total{trailer.weightLimit>0?` / ${trailer.weightLimit}`:""}</div></div>
          <div style={{padding:"8px",borderRadius:"10px",backgroundColor:"#f9fafb",textAlign:"center"}}><div style={{fontSize:"18px",fontWeight:"800",color:"#1f2937"}}>{safety.volPct.toFixed(0)}%</div><div style={{fontSize:"11px",color:"#6b7280"}}>Space</div></div></div>
        {cohVal<K&&<div style={{marginTop:"10px",padding:"10px",backgroundColor:"#fef2f2",borderRadius:"8px",border:"1px solid #fca5a5"}}><p style={{fontSize:"12px",color:"#991b1b",margin:0,fontWeight:"600"}}>✗ INCOHERENT — loading below K=⅔. Compound safety failure. Redistribute weight before travel.</p></div>}</div>

      {doorFail&&doorFail.length>0&&<div style={{...card,borderColor:"#fbbf24",backgroundColor:"#fffbeb"}}><p style={{fontSize:"13px",color:"#92400e",fontWeight:"700",margin:0}}>⚠ Door fit: {doorFail.map(p=>p.item.name).join(", ")}</p></div>}

      {/* Loading step + weight filters */}
      {(()=>{
        const maxN=loadingOrder.length;
        const curStep=Math.min(layerStep,maxN);
        const curItem=loadingOrder.find(p=>p.loadNum===curStep);
        const visIds2=new Set(loadingOrder.filter(p=>p.loadNum<=curStep&&wtVis[p.item.weight]).map(p=>p.item.id));
        // Store visIds on plan object for views to read (avoid recalc)
        plan._visIds=visIds2; plan._curStep=curStep;

        return <div style={card}>
          <div style={{display:"flex",alignItems:"center",gap:"6px",marginBottom:"10px"}}>
            <button onClick={()=>{const ns=Math.max(1,curStep-1);setLayerStep(ns);setSelItem(loadingOrder.find(p2=>p2.loadNum===ns)?.item.id||null);}}
              disabled={curStep<=1} style={{padding:"10px 14px",fontSize:"18px",fontWeight:"700",border:"none",borderRadius:"8px",backgroundColor:curStep>1?"#2563eb":"#e5e7eb",color:curStep>1?"#fff":"#9ca3af",cursor:curStep>1?"pointer":"default"}}>◄</button>
            <div style={{flex:1,textAlign:"center"}}>
              <div style={{fontSize:"15px",fontWeight:"700",color:"#1f2937"}}>{curStep>=maxN?"All "+maxN:"Step "+curStep+" of "+maxN}</div>
              {curItem&&curStep<maxN&&<div style={{fontSize:"12px",color:"#6b7280",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{CAT[curItem.item.category].emoji} {curItem.item.name}</div>}
            </div>
            <button onClick={()=>{const ns=Math.min(maxN,curStep+1);setLayerStep(ns);setSelItem(loadingOrder.find(p2=>p2.loadNum===ns)?.item.id||null);}}
              disabled={curStep>=maxN} style={{padding:"10px 14px",fontSize:"18px",fontWeight:"700",border:"none",borderRadius:"8px",backgroundColor:curStep<maxN?"#2563eb":"#e5e7eb",color:curStep<maxN?"#fff":"#9ca3af",cursor:curStep<maxN?"pointer":"default"}}>►</button>
            <button onClick={()=>{setLayerStep(999);setSelItem(null);}}
              style={{padding:"10px 12px",fontSize:"13px",fontWeight:"700",border:curStep>=maxN?"2px solid #2563eb":"2px solid #d1d5db",borderRadius:"8px",backgroundColor:curStep>=maxN?"#eff6ff":"#fff",color:"#2563eb",cursor:"pointer"}}>All</button>
          </div>
          <div style={{display:"flex",gap:"4px"}}>
            {WT_KEYS.map(k=>{const c=WT[k],on=wtVis[k];return<button key={k} onClick={()=>setWtVis(p=>({...p,[k]:!p[k]}))}
              style={{flex:1,padding:"8px 2px",fontSize:"11px",fontWeight:"700",border:`2px solid ${on?c.color:"#d1d5db"}`,borderRadius:"8px",
                backgroundColor:on?c.color:"#fff",color:on?"#fff":c.color,cursor:"pointer",opacity:on?1:0.5}}>{c.label}</button>;})}
          </div>
        </div>;
      })()}

      <div style={{display:"flex",gap:"4px",marginBottom:"14px",backgroundColor:"#f3f4f6",borderRadius:"10px",padding:"4px"}}>
        {[["tetris","🧱 Top"],["iso","🎲 3D"],["heatmap","🌡️ Heat"],["side","📐 Side"]].map(([k,lb])=><button key={k} onClick={()=>setViewMode(k)}
          style={{flex:1,padding:"10px 4px",fontSize:"13px",fontWeight:"700",border:"none",borderRadius:"8px",cursor:"pointer",backgroundColor:viewMode===k?"#fff":"transparent",color:viewMode===k?"#111":"#6b7280",boxShadow:viewMode===k?"0 1px 3px rgba(0,0,0,0.1)":"none"}}>{lb}</button>)}</div>

      {/* TETRIS — true shape rendering via mask cells */}
      {viewMode==="tetris"&&<div style={card}><p style={{...hint,marginBottom:"8px"}}>Tap item for details. Shapes = actual footprints.</p>
        <div style={{display:"flex",justifyContent:"center",overflowX:"auto"}}>
          <svg width={svgW+pad*2} height={svgH+pad*2+24} viewBox={`0 0 ${svgW+pad*2} ${svgH+pad*2+24}`} style={{maxWidth:"100%"}}>
            <text x={pad+svgW/2} y={14} textAnchor="middle" fontSize="12" fontWeight="700" fill="#374151">▲ FRONT</text>
            <text x={pad+svgW/2} y={svgH+pad*2+14} textAnchor="middle" fontSize="12" fontWeight="700" fill="#374151">▼ BACK (Door)</text>
            <path d={oPath} fill="#f3f4f6" stroke="#374151" strokeWidth="3"/>
            {obsRects.map(r=><rect key={r.key} x={r.x} y={r.y} width={r.w} height={r.h} fill="#9ca3af" opacity={.45} rx="2"/>)}
            <line x1={pad} y1={pad+svgH*.5} x2={pad+svgW} y2={pad+svgH*.5} stroke="#d1d5db" strokeWidth="1" strokeDasharray="6,4"/>
            <line x1={pad+svgW/2} y1={pad} x2={pad+svgW/2} y2={pad+svgH} stroke="#e5e7eb" strokeWidth="1" strokeDasharray="4,4"/>
            {plan.placements.filter(p=>p.position&&plan._visIds&&plan._visIds.has(p.item.id)).map(p=>{
              const pos=p.position,col=CC[p.item.category],isSel=selItem===p.item.id,isStk=pos.z>0;
              const lo=loadingOrder.find(o=>o.item.id===p.item.id);
              const x0=pad+pos.x*sc,y0=pad+pos.y*sc;
              const rects=pos.mask?maskToRects(pos.mask,x0,y0,cellW,cellH):
                [{x:x0,y:y0,w:pos.rw*sc,h:pos.rl*sc}];
              return<g key={p.item.id} onClick={()=>setSelItem(isSel?null:p.item.id)} style={{cursor:"pointer"}}>
                {rects.map((r,ri)=><rect key={ri} x={r.x+.5} y={r.y+.5} width={r.w-1} height={r.h-1} fill={col} opacity={isStk?.65:.88} stroke={isSel?"#fff":"rgba(0,0,0,0.1)"} strokeWidth={isSel?2:.5} rx="1"/>)}
                <text x={x0+pos.gw*cellW/2} y={y0+pos.gl*cellH/2} textAnchor="middle" dominantBaseline="central" fontSize={Math.min(13,Math.min(pos.gw*cellW,pos.gl*cellH)*.4)} fontWeight="800" fill="#fff">{lo?lo.loadNum:"?"}</text>
              </g>;})}
          </svg></div>
        <div style={{display:"flex",flexWrap:"wrap",gap:"8px",marginTop:"8px",justifyContent:"center"}}>
          {CAT_KEYS.map(k=><div key={k} style={{display:"flex",alignItems:"center",gap:"3px"}}><div style={{width:12,height:12,borderRadius:3,backgroundColor:CC[k]}}/><span style={{fontSize:"11px",color:"#6b7280"}}>{CAT[k].emoji} {CAT[k].label}</span></div>)}</div></div>}

      {/* 3D ISOMETRIC VIEW */}
      {viewMode==="iso"&&(()=>{
        const IC=0.866,IS=0.5; // cos30, sin30
        const iW=(trailer.w+trailer.l)*IC,iH2=(trailer.w+trailer.l)*IS+trailer.h;
        const iSc=Math.min(360/iW,340/iH2);
        const svgIW=iW*iSc+60,svgIH=iH2*iSc+60;
        const ox=30+trailer.l*IC*iSc,oy=trailer.h*iSc+20;
        const ip=(x2,y2,z2)=>[(x2-y2)*IC*iSc+ox,(x2+y2)*IS*iSc-z2*iSc+oy];
        const pp=(pts)=>pts.map(p2=>`${p2[0]},${p2[1]}`).join(" ");

        // Sort items: far-from-viewer first (low x+y drawn first)
        const sorted=plan.placements.filter(p=>p.position&&plan._visIds&&plan._visIds.has(p.item.id))
          .map(p=>({...p})).sort((a,b)=>(a.position.x+a.position.y+a.position.z)-(b.position.x+b.position.y+b.position.z));

        const W=trailer.w,L=trailer.l,H=trailer.h;
        // ═══ Profile-accurate floor ═══
        const _cs=(trailer.cs&&trailer.cs.length>=2)?[...trailer.cs].sort((a,b)=>a.h-b.h):[{h:0,w:W},{h:H,w:W}];
        const _lp=(trailer.lp&&trailer.lp.length>=1)?[...trailer.lp].sort((a,b)=>a.d-b.d):[{d:0,w:W},{d:L,w:W}];
        const _fW=_cs[0].w;const _flPts=[];const _fS=30;
        for(let i=0;i<=_fS;i++){const y2=(i/_fS)*L;const wAtY=interp(_lp,y2,"d","w");const sw=_fW>0?Math.min(wAtY/_fW,1)*_fW:0;const cx2=W/2;_flPts.push({l:ip(cx2-sw/2,y2,0),r:ip(cx2+sw/2,y2,0)});}
        let floorPath=`M ${_flPts[0].l[0]},${_flPts[0].l[1]}`;for(const fp of _flPts)floorPath+=` L ${fp.l[0]},${fp.l[1]}`;
        floorPath+=` L ${_flPts[_flPts.length-1].r[0]},${_flPts[_flPts.length-1].r[1]}`;for(let i=_flPts.length-1;i>=0;i--)floorPath+=` L ${_flPts[i].r[0]},${_flPts[i].r[1]}`;floorPath+=" Z";

        return <div style={card}><p style={{...hint,marginBottom:"8px"}}>Isometric 3D. Tap item for details.</p>
          <div style={{display:"flex",justifyContent:"center",overflowX:"auto"}}>
            <svg width={svgIW} height={svgIH} viewBox={`0 0 ${svgIW} ${svgIH}`} style={{maxWidth:"100%"}}>
              {/* Trailer shell — floor, back wall, left wall */}
              <path d={floorPath} fill="#e5e7eb" stroke="#9ca3af" strokeWidth="1"/>
              <polygon points={pp([ip(0,L,0),ip(W,L,0),ip(W,L,H),ip(0,L,H)])} fill="#d1d5db" stroke="#9ca3af" strokeWidth="1"/>
              <polygon points={pp([ip(0,0,0),ip(0,L,0),ip(0,L,H),ip(0,0,H)])} fill="#e5e7eb" stroke="#9ca3af" strokeWidth="1"/>

              {/* Obstacle boxes */}
              {(trailer.obs||[]).map((o,oi)=>{
                const ox2=o.x,oy2=o.y;
                return <g key={`obs${oi}`}>
                  <polygon points={pp([ip(ox2,oy2,0),ip(ox2+o.ow,oy2,0),ip(ox2+o.ow,oy2+o.ol,0),ip(ox2,oy2+o.ol,0)])} fill="#9ca3af" opacity={0.5}/>
                  <polygon points={pp([ip(ox2,oy2,o.oh),ip(ox2+o.ow,oy2,o.oh),ip(ox2+o.ow,oy2+o.ol,o.oh),ip(ox2,oy2+o.ol,o.oh)])} fill="#9ca3af" opacity={0.35}/>
                </g>;
              })}

              {/* Items as isometric boxes */}
              {sorted.map(p=>{
                const pos=p.position,col=CC[p.item.category];
                const ix=pos.x,iy=pos.y,iz=pos.z,iw=pos.rw,il=pos.rl,ih=p.item.h;
                const isSel=selItem===p.item.id;
                const lo=loadingOrder.find(o2=>o2.item.id===p.item.id);
                const stk=isSel?"#fff":"rgba(0,0,0,0.15)";const sw=isSel?2.5:0.5;
                return <g key={p.item.id} onClick={()=>setSelItem(isSel?null:p.item.id)} style={{cursor:"pointer"}}>
                  {/* Right face */}
                  <polygon points={pp([ip(ix+iw,iy,iz),ip(ix+iw,iy+il,iz),ip(ix+iw,iy+il,iz+ih),ip(ix+iw,iy,iz+ih)])} fill={shade(col,0.7)} stroke={stk} strokeWidth={sw}/>
                  {/* Front face */}
                  <polygon points={pp([ip(ix,iy,iz),ip(ix+iw,iy,iz),ip(ix+iw,iy,iz+ih),ip(ix,iy,iz+ih)])} fill={shade(col,0.85)} stroke={stk} strokeWidth={sw}/>
                  {/* Top face */}
                  <polygon points={pp([ip(ix,iy,iz+ih),ip(ix+iw,iy,iz+ih),ip(ix+iw,iy+il,iz+ih),ip(ix,iy+il,iz+ih)])} fill={col} stroke={stk} strokeWidth={sw}/>
                  {/* Label */}
                  {(()=>{const c2=ip(ix+iw/2,iy+il/2,iz+ih);return <text x={c2[0]} y={c2[1]-2} textAnchor="middle" fontSize="10" fontWeight="800" fill="#fff" style={{pointerEvents:"none"}}>{lo?lo.loadNum:"?"}</text>;})()}
                </g>;
              })}

              {/* Trailer wireframe edges (front/right/top) */}
              <polyline points={pp([ip(W,0,0),ip(W,0,H),ip(0,0,H)])} fill="none" stroke="#374151" strokeWidth="2" strokeDasharray="6,3"/>
              <polyline points={pp([ip(W,0,0),ip(W,L,0)])} fill="none" stroke="#374151" strokeWidth="2"/>
              <polyline points={pp([ip(W,0,H),ip(W,L,H),ip(0,L,H)])} fill="none" stroke="#374151" strokeWidth="1.5" strokeDasharray="4,3"/>
              <line x1={ip(0,0,H)[0]} y1={ip(0,0,H)[1]} x2={ip(0,0,0)[0]} y2={ip(0,0,0)[1]} stroke="#374151" strokeWidth="2"/>

              {/* Labels */}
              {(()=>{const fl=ip(W/2,0,H+4);return <text x={fl[0]} y={fl[1]-6} textAnchor="middle" fontSize="11" fontWeight="700" fill="#374151">FRONT</text>;})()}
              {(()=>{const bl=ip(W/2,L,H+4);return <text x={bl[0]} y={bl[1]-6} textAnchor="middle" fontSize="11" fontWeight="700" fill="#374151">BACK</text>;})()}
            </svg>
          </div>
          <div style={{display:"flex",flexWrap:"wrap",gap:"8px",marginTop:"8px",justifyContent:"center"}}>
            {CAT_KEYS.map(k=><div key={k} style={{display:"flex",alignItems:"center",gap:"3px"}}><div style={{width:12,height:12,borderRadius:3,backgroundColor:CC[k]}}/><span style={{fontSize:"11px",color:"#6b7280"}}>{CAT[k].emoji} {CAT[k].label}</span></div>)}</div>
        </div>;
      })()}

      {viewMode==="heatmap"&&heatmap&&<div style={card}><p style={{...hint,marginBottom:"8px"}}>Blue=light Red=heavy. Guided placement live.</p>
        <div style={{display:"flex",justifyContent:"center",overflowX:"auto"}}>
          <svg width={svgW+pad*2} height={svgH+pad*2+24} viewBox={`0 0 ${svgW+pad*2} ${svgH+pad*2+24}`} style={{maxWidth:"100%"}}>
            <text x={pad+svgW/2} y={14} textAnchor="middle" fontSize="12" fontWeight="700" fill="#374151">▲ FRONT</text>
            <text x={pad+svgW/2} y={svgH+pad*2+14} textAnchor="middle" fontSize="12" fontWeight="700" fill="#374151">▼ BACK</text>
            <path d={oPath} fill="#1e293b" stroke="#374151" strokeWidth="3"/>
            {heatmap.map((row,ri)=>row.map((val,ci)=>{if(val<=0)return null;return<rect key={`${ri}-${ci}`} x={pad+ci*HEAT*sc} y={pad+ri*HEAT*sc} width={Math.min(HEAT*sc,svgW-ci*HEAT*sc)} height={Math.min(HEAT*sc,svgH-ri*HEAT*sc)} fill={heatColor(val,heatMax)} opacity={.85}/>;}))}
            {obsRects.map(r=><rect key={`h${r.key}`} x={r.x} y={r.y} width={r.w} height={r.h} fill="#000" opacity={.3}/>)}
          </svg></div>
        <div style={{display:"flex",alignItems:"center",gap:"4px",justifyContent:"center",marginTop:"8px"}}>
          <span style={hint}>Light</span>{HS.map(([,col],i)=><div key={i} style={{width:28,height:12,backgroundColor:col,borderRadius:i===0?"4px 0 0 4px":i===HS.length-1?"0 4px 4px 0":"0"}}/>)}<span style={hint}>Heavy</span></div></div>}

      {viewMode==="side"&&<div style={card}><p style={{...hint,marginBottom:"8px"}}>Side profile. Left=front.</p>
        {(()=>{const sideSc=Math.min(mxW/trailer.l,200/trailer.h),ssW=trailer.l*sideSc,ssH=trailer.h*sideSc;
          return<div style={{display:"flex",justifyContent:"center",overflowX:"auto"}}>
            <svg width={ssW+pad*2} height={ssH+pad+20} viewBox={`0 0 ${ssW+pad*2} ${ssH+pad+20}`} style={{maxWidth:"100%"}}>
              <text x={pad+4} y={12} fontSize="11" fontWeight="700" fill="#374151">Front</text><text x={pad+ssW-4} y={12} fontSize="11" fontWeight="700" fill="#374151" textAnchor="end">Back</text>
              <rect x={pad} y={18} width={ssW} height={ssH} fill="#f3f4f6" stroke="#374151" strokeWidth="2" rx="3"/>
              <line x1={pad} y1={18+ssH} x2={pad+ssW} y2={18+ssH} stroke="#374151" strokeWidth="3"/>
              {safety.cogHeight>0&&<><line x1={pad} y1={18+ssH-safety.cogHeight*sideSc} x2={pad+ssW} y2={18+ssH-safety.cogHeight*sideSc} stroke="#dc2626" strokeWidth="1.5" strokeDasharray="5,3"/><text x={pad+ssW+4} y={18+ssH-safety.cogHeight*sideSc+4} fontSize="9" fill="#dc2626">CoG</text></>}
              {plan.placements.filter(p=>p.position&&plan._visIds&&plan._visIds.has(p.item.id)).map(si=>{const pos=si.position;return<rect key={si.item.id} x={pad+pos.y*sideSc} y={18+ssH-(pos.z+si.item.h)*sideSc} width={Math.max(pos.rl*sideSc,2)} height={Math.max(si.item.h*sideSc,2)} fill={CC[si.item.category]} opacity={.8} stroke="rgba(0,0,0,0.15)" strokeWidth="1" rx="2" onClick={()=>setSelItem(selItem===si.item.id?null:si.item.id)} style={{cursor:"pointer"}}/>;})}
            </svg></div>;})()}</div>}

      {selItem&&(()=>{const p=plan.placements.find(pp=>pp.item.id===selItem);if(!p||!p.position)return null;const wc=WT[p.item.weight],cc2=CAT[p.item.category],lo=loadingOrder.find(o=>o.item.id===selItem);
        return<div style={{...card,borderLeft:`5px solid ${CC[p.item.category]}`,padding:"14px 16px"}}><div style={{fontWeight:"700",fontSize:"16px",color:"#1f2937",marginBottom:"4px"}}>{cc2.emoji} {p.item.name}</div>
          <div style={{fontSize:"13px",color:"#6b7280",lineHeight:1.8}}>{fmtD3(p.item.l,p.item.w,p.item.h)} · {p.item.exactLbs?`${p.item.exactLbs} lbs`:wc.label}{p.item.surface==="round"?" · round":""}{p.item.cutouts?.length?` · ${p.item.cutouts.length} cutout${p.item.cutouts.length>1?"s":""}`:""}<br/>
            {zoneLabel(p.position,trailer)}<br/>Load #{lo?lo.loadNum:"?"} of {loadingOrder.length}</div></div>;})()}

      {unplaced.length>0&&<div style={{...card,borderColor:"#fca5a5",backgroundColor:"#fef2f2"}}><h3 style={{fontSize:"15px",fontWeight:"700",color:"#991b1b",marginBottom:"8px"}}>✗ {unplaced.length} won't fit</h3>
        {unplaced.map(p=><div key={p.item.id} style={{fontSize:"13px",color:"#991b1b",marginBottom:"2px"}}>• {CAT[p.item.category].emoji} {p.item.name}</div>)}</div>}

      <div style={card}><h2 style={{fontSize:"16px",fontWeight:"700",marginBottom:"4px",color:"#1f2937"}}>Loading Order</h2><p style={{...hint,marginBottom:"10px"}}>#1 first → pushed to front.</p>
        {loadingOrder.map(p=>{const cc2=CAT[p.item.category],wc=WT[p.item.weight];const isVis=plan._visIds&&plan._visIds.has(p.item.id);return<div key={p.item.id} onClick={()=>setSelItem(selItem===p.item.id?null:p.item.id)}
          style={{display:"flex",alignItems:"center",gap:"10px",padding:"10px 12px",marginBottom:"5px",borderRadius:"10px",cursor:"pointer",backgroundColor:selItem===p.item.id?"#eff6ff":"#f9fafb",border:selItem===p.item.id?"2px solid #2563eb":"1px solid #e5e7eb",opacity:isVis?1:0.35,transition:"opacity 0.2s"}}>
          <div style={{width:30,height:30,borderRadius:"50%",backgroundColor:CC[p.item.category],color:"#fff",display:"flex",alignItems:"center",justifyContent:"center",fontWeight:"800",fontSize:"13px",flexShrink:0}}>{p.loadNum}</div>
          <div style={{flex:1,minWidth:0}}><div style={{fontWeight:"700",fontSize:"13px",color:"#1f2937",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{cc2.emoji} {p.item.name}</div>
            <div style={{fontSize:"11px",color:"#6b7280"}}>{zoneLabel(p.position,trailer)} · {p.item.exactLbs?`${p.item.exactLbs} lbs`:wc.label}</div></div></div>;})}</div>

      <button onClick={()=>{setPlan(null);setStep("items");setSelItem(null);setErr("");}} style={{...btnP,marginBottom:"10px"}}>← Add More & Recalculate</button>
      <button onClick={manualSave} disabled={!storageOk} style={{...btnS,marginBottom:"10px",opacity:storageOk?1:0.4}}>{!storageOk?"💾 N/A":savedMsg||"💾 Save"}</button>
      <button onClick={()=>{if(window.confirm("Reset everything?")){setStep("trailer");setTrailer(null);setItems([]);setPlan(null);setSelItem(null);setTFt({l:"",w:"",h:""});setTIn({l:"",w:"",h:""});setWLim("");setCsRows([]);setLpRows([]);setObsRows([]);setDoorWIn("");setDoorHIn("");setCExact("");setUndoStack([]);setErr("");setShowAdv(false);(async()=>{try{await window.storage.delete('tlp-auto');await window.storage.delete('tlp-manual');}catch(e){}})();}}}
        style={{...btnS,backgroundColor:"#dc2626",marginBottom:"20px"}}>Reset All</button>
    </div>);}
  return null;
}
