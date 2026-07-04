"""
ski_expert.py — 双板滑雪专家系统(第五关)
输入:每帧物理量(经ski_physics) + 阶段标签 + 内倾曲线
输出:带置信度的技术分析报告
依赖:ski_physics.py (同目录)
"""
import numpy as np
from collections import defaultdict
import ski_physics
from ski_physics import extract_all

# ========================================================
# 数据质量门控
# ========================================================
def assess_data_quality(seq, iids, trunk_s, apexes):
    T = len(iids); report = {'warnings': [], 'metrics': {}}
    tc = {'内倾':0, '重心':0, '膝角':0, '反弓':0}
    for iid in iids:
        r = extract_all(seq[iid]['annotation'])
        if r['trunk_tilt'][1]: tc['内倾'] += 1
        if r['com'][3]: tc['重心'] += 1
        if r['joints']['右膝'][1] and r['joints']['左膝'][1]: tc['膝角'] += 1
        if r['angulation_R'][1]: tc['反弓'] += 1
    for k, v in tc.items():
        rate = v / T; report['metrics'][f'{k}可信率'] = rate
        if rate < 0.5: report['warnings'].append(f'⚠️ {k}可信率仅{rate:.0%},遮挡严重')
    signs = [np.sign(trunk_s[i]) for i, _ in apexes]
    if signs:
        pos = sum(1 for s in signs if s > 0); neg = sum(1 for s in signs if s < 0)
        tot = len(signs); mr = min(pos, neg)/tot if tot else 0
        report['metrics']['左右弯平衡度'] = mr
        if mr < 0.2:
            report['warnings'].append(
                f'⚠️ 转弯几乎全同方向({max(pos,neg)}/{tot})。可能:(a)视角问题使内倾方向失真;'
                f'或(b)真实同向弯为主。建议确认拍摄角度或换侧后方、相机水平的视频。'
                f'内倾【方向】置信度低,内倾【幅度】仍可参考。')
            report['viewpoint_ok'] = False
        else:
            report['viewpoint_ok'] = True
    mt = np.median(trunk_s); report['metrics']['内倾中位数'] = mt
    if abs(mt) > 8:
        report['warnings'].append(f'⚠️ 内倾中位数{mt:+.1f}°偏离0,存在基线偏移,内倾方向需谨慎。')
    report['overall'] = '低' if not report.get('viewpoint_ok', True) else ('中' if len(report['warnings'])>=2 else '高')
    return report

# ========================================================
# 16条规则(每条返回统一格式dict)
# ========================================================
def rule_aframe(seq, iids, labels, trunk_s):
    KR,KL,AR,AL = 11,14,12,15
    pa = defaultdict(list)
    for i, iid in enumerate(iids):
        a = seq[iid]['annotation']
        kR,kL,aR,aL = [np.asarray(a[x],float) for x in [KR,KL,AR,AL]]
        if any(p[2]<1 for p in [kR,kL,aR,aL]): continue
        kd=abs(kL[0]-kR[0]); fd=abs(aL[0]-aR[0])
        if kd<1e-6: continue
        pa[labels[i]].append(fd/kd)
    ini=np.mean(pa['入弯']) if pa['入弯'] else np.nan
    com=np.mean(pa['出弯']) if pa['出弯'] else np.nan
    if np.isnan(ini) or np.isnan(com):
        return {'触发':False,'建议':'','证据':'数据不足','依赖':['膝踝距离'],'规则置信度':'中'}
    d=ini-com; t=(ini>1.2 and d>0.15)
    return {'触发':t,'建议':'入弯时脚比膝明显偏开,可能存在轻度A-frame,提示内侧腿不够主动内倾。可用内侧脚踝引导倾倒,让内侧膝更早跟上。' if t else '',
            '证据':f'入弯A-frame指标{ini:.2f} vs 出弯{com:.2f}(差{d:+.2f})','依赖':['膝踝距离'],'规则置信度':'中'}

def rule_inner_leg(seq, iids, labels, trunk_s, front_view=True):
    diffs=[]
    for i in range(len(iids)):
        if labels[i]!='顶点': continue
        r=extract_all(seq[iids[i]]['annotation'])
        kr,okr=r['joints']['右膝']; kl,okl=r['joints']['左膝']
        if not(okr and okl): continue
        if not(90<=kr<=180 and 90<=kl<=180): continue
        inner,outer=(kr,kl) if (trunk_s[i]<0)==front_view else (kl,kr)
        diffs.append(inner-outer)
    if not diffs:
        return {'触发':False,'建议':'','证据':'数据不足','依赖':['膝角','内倾方向'],'规则置信度':'低'}
    m=np.median(diffs); t=(m>=-3)
    return {'触发':t,'建议':'顶点内侧腿屈曲偏少,应主动深屈收短,为身体内倾让空间、建立更大刃角。' if t else '',
            '证据':f'内侧-外侧膝角中位数{m:+.0f}°(共{len(diffs)}顶点)','依赖':['膝角','内倾方向'],'规则置信度':'低'}

def rule_fore_aft(seq, iids, labels, trunk_s):
    AR,AL=12,15; po={'入弯':[],'顶点':[],'出弯':[]}
    for i in range(len(iids)):
        if labels[i] not in po: continue
        r=extract_all(seq[iids[i]]['annotation'])
        cx,cy,mc,ok=r['com']
        if not ok: continue
        a=seq[iids[i]]['annotation']; aR,aL=np.asarray(a[AR],float),np.asarray(a[AL],float)
        if aR[2]<1 or aL[2]<1: continue
        po[labels[i]].append(cx-(aR[0]+aL[0])/2)
    mn={k:(np.median(v) if v else np.nan) for k,v in po.items()}
    if np.isnan(mn['入弯']) or np.isnan(mn['出弯']):
        return {'触发':False,'建议':'','证据':'重心/阶段数据不足','依赖':['重心','阶段标签'],'规则置信度':'低'}
    sh=abs(mn['入弯']-mn['出弯']); t=(sh<0.02)
    return {'触发':t,'建议':'入弯与出弯重心位置变化很小,前后转移可能不足。健康转弯中重心应随阶段流动。' if t else '',
            '证据':f'重心偏移 入弯{mn["入弯"]:+.3f} 顶点{mn["顶点"]:+.3f} 出弯{mn["出弯"]:+.3f}(变化{sh:.3f})',
            '依赖':['重心','阶段标签'],'规则置信度':'低'}

def rule_symmetry(seq, iids, labels, trunk_s, front_view=True):
    lt={'反弓':[],'内倾':[]}; rt={'反弓':[],'内倾':[]}
    for i in range(len(iids)):
        if labels[i]!='顶点': continue
        r=extract_all(seq[iids[i]]['annotation']); vals=[]
        if r['angulation_R'][1]: vals.append(abs(r['angulation_R'][0]))
        if r['angulation_L'][1]: vals.append(abs(r['angulation_L'][0]))
        vals=[v for v in vals if v<=60]; angu=max(vals) if vals else None
        g=rt if ((trunk_s[i]<0)==front_view) else lt
        if angu is not None: g['反弓'].append(angu)
        g['内倾'].append(abs(trunk_s[i]))
    nl,nr=len(lt['内倾']),len(rt['内倾'])
    if nl<2 or nr<2:
        return {'触发':False,'建议':'','证据':f'左弯{nl}/右弯{nr},一侧不足','依赖':['反弓','内倾方向'],'规则置信度':'低'}
    la,ra=np.median(lt['反弓']) if lt['反弓'] else np.nan, np.median(rt['反弓']) if rt['反弓'] else np.nan
    li,ri=np.median(lt['内倾']),np.median(rt['内倾'])
    ad=abs(la-ra) if not(np.isnan(la)or np.isnan(ra)) else np.nan; idf=abs(li-ri)
    t=(not np.isnan(ad) and ad>10) or (idf>8); w='左弯' if li<ri else '右弯'
    return {'触发':t,'建议':f'左右转弯不对称({w}偏弱),建议针对性练习较弱侧。' if t else '',
            '证据':f'左弯(反弓{la:.0f}/内倾{li:.0f}) vs 右弯(反弓{ra:.0f}/内倾{ri:.0f})','依赖':['反弓','内倾方向'],'规则置信度':'中'}

def rule_angulation(seq, iids, labels):
    aa=[]
    for i in range(len(iids)):
        if labels[i]!='顶点': continue
        r=extract_all(seq[iids[i]]['annotation']); vals=[]
        if r['angulation_R'][1]: vals.append(abs(r['angulation_R'][0]))
        if r['angulation_L'][1]: vals.append(abs(r['angulation_L'][0]))
        vals=[v for v in vals if v<=60]
        if vals: aa.append(max(vals))
    if not aa:
        return {'触发':False,'建议':'','证据':'顶点无可信反弓','依赖':['反弓'],'规则置信度':'中'}
    m=np.median(aa); t=(m<10)
    return {'触发':t,'建议':'顶点外侧腿反弓偏小,可能靠整体倒向弯内而非上下身分离。可上身更直立、下肢独立内倾。' if t else '',
            '证据':f'顶点外侧腿反弓中位数{m:.0f}°(共{len(aa)}顶点,阈值10°)','依赖':['反弓'],'规则置信度':'中'}

def rule_stance_width(seq, iids, labels):
    HR,HL,AR,AL=10,13,12,15; rs=[]
    for i in range(len(iids)):
        if labels[i]!='过渡': continue
        a=seq[iids[i]]['annotation']; hR,hL,aR,aL=[np.asarray(a[x],float) for x in [HR,HL,AR,AL]]
        if any(p[2]<1 for p in [hR,hL,aR,aL]): continue
        hw=abs(hL[0]-hR[0]); fw=abs(aL[0]-aR[0])
        if hw<1e-6: continue
        rt=fw/hw
        if rt<=5: rs.append(rt)
    if not rs:
        return {'触发':False,'建议':'','证据':'过渡阶段无数据','依赖':['脚髋距离'],'规则置信度':'中'}
    m=np.median(rs); t=(m>1.4)
    return {'触发':t,'建议':f'过渡站姿偏宽(脚约髋宽{m:.1f}倍),会限制换刃灵活性。可收窄至与髋同宽。' if t else '',
            '证据':f'过渡 脚间距/髋宽 中位数{m:.2f}(共{len(rs)}帧,阈值1.4)','依赖':['脚髋距离'],'规则置信度':'中'}

def rule_rhythm(seq, iids, labels, trunk_s, apexes):
    af=sorted([i for i,_ in apexes]); note=''
    if len(af)>=3:
        iv=np.diff(af); cv=np.std(iv)/np.mean(iv); note=f'间隔{iv.tolist()},变异{cv:.2f}'; rb=(cv>0.6)
    else: cv=np.nan; rb=False; note=f'顶点不足({len(af)})'
    d2=np.diff(trunk_s,2); rough=np.sqrt(np.mean(d2**2)); rgb=(rough>2.0)
    t=rb or rgb; rs=[]
    if rb: rs.append('节奏不均匀')
    if rgb: rs.append('内倾变化不平滑')
    return {'触发':t,'建议':f'流畅度可提升({"、".join(rs)})。应连贯过渡、节奏均匀。' if t else '',
            '证据':f'{note};粗糙度{rough:.2f}','依赖':['内倾幅度'],'规则置信度':'中'}

def rule_vertical_separation(seq, iids, labels):
    AR,AL,HR,HL=12,15,10,13; rs=[]
    for i in range(len(iids)):
        if labels[i]!='顶点': continue
        a=seq[iids[i]]['annotation']; aR,aL,hR,hL=[np.asarray(a[x],float) for x in [AR,AL,HR,HL]]
        if any(p[2]<1 for p in [aR,aL,hR,hL]): continue
        vs=abs(aL[1]-aR[1]); hw=abs(hL[0]-hR[0])
        if hw<1e-6: continue
        rt=vs/hw
        if rt<=5: rs.append(rt)
    if not rs:
        return {'触发':False,'建议':'','证据':'顶点无数据','依赖':['脚髋距离'],'规则置信度':'中'}
    m=np.median(rs); t=(m<0.3)
    return {'触发':t,'建议':f'顶点垂直分离偏小(高低差约髋宽{m:.1f}倍),内外腿屈伸不足。可加强一屈一伸。' if t else '',
            '证据':f'顶点 高低差/髋宽 中位数{m:.2f}(共{len(rs)}帧,阈值0.3)','依赖':['脚髋距离'],'规则置信度':'中'}

def rule_flexion(seq, iids, labels):
    py={'顶点':[],'过渡':[]}; ay=[]
    for i in range(len(iids)):
        r=extract_all(seq[iids[i]]['annotation']); cx,cy,mc,ok=r['com']
        if not ok: continue
        ay.append(cy)
        if labels[i] in py: py[labels[i]].append(cy)
    if len(ay)<5:
        return {'触发':False,'建议':'','证据':'重心数据不足','依赖':['重心','阶段标签'],'规则置信度':'低'}
    yr=np.max(ay)-np.min(ay); apy=np.median(py['顶点']) if py['顶点'] else np.nan; try_=np.median(py['过渡']) if py['过渡'] else np.nan
    t=(yr<0.05); dn=''
    if not(np.isnan(apy)or np.isnan(try_)): dn='顶点低/过渡高,方向正确' if apy>try_ else '顶点未明显低于过渡,屈伸不清晰'
    return {'触发':t,'建议':f'屈伸幅度偏小(起伏{yr:.2f}),腿部主动屈伸不足。应顶点屈、过渡伸,规律起伏。' if t else '',
            '证据':f'重心高度起伏{yr:.2f}(阈值0.05);{dn}','依赖':['重心','阶段标签'],'规则置信度':'中'}

def rule_inclination(seq, iids, labels, trunk_s):
    ai=[abs(trunk_s[i]) for i in range(len(iids)) if labels[i]=='顶点']
    if not ai:
        return {'触发':False,'建议':'','证据':'无顶点数据','依赖':['内倾幅度'],'规则置信度':'中'}
    m=np.median(ai); t=(m<12)
    return {'触发':t,'建议':f'顶点内倾偏小({m:.0f}°),立刃可能不足。可让身体更多倒向弯心。' if t else '',
            '证据':f'顶点内倾中位数{m:.0f}°(阈值12°,共{len(ai)}顶点)','依赖':['内倾幅度'],'规则置信度':'中'}

def rule_transition(seq, iids, labels):
    nt=sum(1 for l in labels if l=='过渡'); n=len(labels); r=nt/n; t=(r>0.35)
    return {'触发':t,'建议':f'过渡占比偏高({r:.0%}),换刃可能拖沓。应快速换刃。' if t else '',
            '证据':f'过渡帧占比{r:.0%}({nt}/{n},阈值35%)','依赖':['阶段标签'],'规则置信度':'中'}

def rule_angulation_phased(seq, iids, labels):
    pa={'入弯':[],'顶点':[]}
    for i in range(len(iids)):
        if labels[i] not in pa: continue
        r=extract_all(seq[iids[i]]['annotation']); vals=[]
        if r['angulation_R'][1]: vals.append(abs(r['angulation_R'][0]))
        if r['angulation_L'][1]: vals.append(abs(r['angulation_L'][0]))
        vals=[v for v in vals if v<=60]
        if vals: pa[labels[i]].append(max(vals))
    ai=np.median(pa['入弯']) if pa['入弯'] else np.nan; ap=np.median(pa['顶点']) if pa['顶点'] else np.nan
    if np.isnan(ai) or np.isnan(ap):
        return {'触发':False,'建议':'','证据':'入弯/顶点反弓不足','依赖':['反弓','阶段标签'],'规则置信度':'中'}
    t=(ap<=ai)
    return {'触发':t,'建议':f'反弓在顶点未明显增大(入弯{ai:.0f}→顶点{ap:.0f}),应随过弯建立、顶点最大。' if t else '',
            '证据':f'反弓 入弯{ai:.0f}°→顶点{ap:.0f}°','依赖':['反弓','阶段标签'],'规则置信度':'中'}

def rule_consistency(seq, iids, labels, trunk_s, apexes):
    ai=[abs(trunk_s[i]) for i,_ in apexes]
    if len(ai)<3:
        return {'触发':False,'建议':'','证据':'顶点不足','依赖':['内倾幅度'],'规则置信度':'中'}
    cv=np.std(ai)/np.mean(ai); t=(cv>0.4)
    return {'触发':t,'建议':f'各弯倾斜不一致(变异{cv:.2f}),质量不稳定。追求每弯一致。' if t else '',
            '证据':f'顶点内倾变异{cv:.2f}(共{len(ai)}弯,阈值0.4)','依赖':['内倾幅度'],'规则置信度':'中'}

def rule_knee_flexion(seq, iids, labels):
    ks=[]
    for i in range(len(iids)):
        if labels[i]!='顶点': continue
        r=extract_all(seq[iids[i]]['annotation'])
        for k in ['右膝','左膝']:
            a,ok=r['joints'][k]
            if ok and 90<=a<=180: ks.append(a)
    if not ks:
        return {'触发':False,'建议':'','证据':'膝角不足','依赖':['膝角'],'规则置信度':'中'}
    m=np.median(ks); t=(m>165)
    return {'触发':t,'建议':f'顶点膝偏直({m:.0f}°),屈曲不足难承压。加强屈膝。' if t else '',
            '证据':f'顶点膝角中位数{m:.0f}°(阈值165°)','依赖':['膝角'],'规则置信度':'中'}

def rule_com_lateral(seq, iids, labels):
    xs=[]
    for i in range(len(iids)):
        r=extract_all(seq[iids[i]]['annotation']); cx,cy,mc,ok=r['com']
        if ok: xs.append(cx)
    if len(xs)<5:
        return {'触发':False,'建议':'','证据':'重心不足','依赖':['重心'],'规则置信度':'中'}
    xr=np.max(xs)-np.min(xs); t=(xr<0.1)
    return {'触发':t,'建议':f'重心横移偏小({xr:.2f}),换弯cross-over可能不足。加强重心横移。' if t else '',
            '证据':f'重心横移范围{xr:.2f}(阈值0.1)','依赖':['重心'],'规则置信度':'中'}

def rule_incl_angu_ratio(seq, iids, labels, trunk_s):
    rs=[]
    for i in range(len(iids)):
        if labels[i]!='顶点': continue
        r=extract_all(seq[iids[i]]['annotation']); vals=[]
        if r['angulation_R'][1]: vals.append(abs(r['angulation_R'][0]))
        if r['angulation_L'][1]: vals.append(abs(r['angulation_L'][0]))
        vals=[v for v in vals if v<=60]
        if not vals: continue
        angu=max(vals); incl=abs(trunk_s[i])
        if incl<1: continue
        rs.append(angu/incl)
    if not rs:
        return {'触发':False,'建议':'','证据':'数据不足','依赖':['反弓','内倾幅度'],'规则置信度':'中'}
    m=np.median(rs); t=(m<0.5)
    return {'触发':t,'建议':f'过弯主要靠整体倒(反弓/内倾{m:.1f}),反弓占比低。加强上下身分离。' if t else '',
            '证据':f'反弓/内倾比中位数{m:.1f}(阈值0.5)','依赖':['反弓','内倾幅度'],'规则置信度':'中'}

# ========================================================
# 置信度传导 + 主引擎
# ========================================================
def combine_confidence(rule_conf, deps, quality):
    lv={'高':3,'中':2,'低':1}; f=lv[rule_conf]; notes=[]
    if '内倾方向' in deps and not quality.get('viewpoint_ok',True):
        f=min(f,1); notes.append('视角问题使内倾方向不可信')
    if '阶段标签' in deps and abs(quality['metrics'].get('内倾中位数',0))>8:
        f=min(f,2); notes.append('基线偏移使阶段标签欠可靠')
    dm={'膝角':'膝角可信率','膝踝距离':'膝角可信率','脚髋距离':'膝角可信率','重心':'重心可信率','反弓':'反弓可信率'}
    for d in deps:
        if d in dm and quality['metrics'].get(dm[d],1.0)<0.5:
            f=min(f,1); notes.append(f'{d}遮挡严重')
    return {3:'高',2:'中',1:'低'}[f], notes

def run_expert_system(seq, iids, labels, trunk_s, apexes, quality, verbose=True):
    rules=[
        ('A-frame检测', rule_aframe(seq,iids,labels,trunk_s)),
        ('内侧腿屈曲', rule_inner_leg(seq,iids,labels,trunk_s)),
        ('前后重心分阶段', rule_fore_aft(seq,iids,labels,trunk_s)),
        ('左右对称性', rule_symmetry(seq,iids,labels,trunk_s)),
        ('外侧腿反弓', rule_angulation(seq,iids,labels)),
        ('站姿宽度', rule_stance_width(seq,iids,labels)),
        ('动作流畅度', rule_rhythm(seq,iids,labels,trunk_s,apexes)),
        ('垂直分离', rule_vertical_separation(seq,iids,labels)),
        ('屈伸循环', rule_flexion(seq,iids,labels)),
        ('内倾充分性', rule_inclination(seq,iids,labels,trunk_s)),
        ('换刃过渡速度', rule_transition(seq,iids,labels)),
        ('反弓分阶段', rule_angulation_phased(seq,iids,labels)),
        ('转弯一致性', rule_consistency(seq,iids,labels,trunk_s,apexes)),
        ('膝屈曲水平', rule_knee_flexion(seq,iids,labels)),
        ('重心横移', rule_com_lateral(seq,iids,labels)),
        ('内倾反弓配比', rule_incl_angu_ratio(seq,iids,labels,trunk_s)),
    ]
    for name,res in rules:
        res['_conf'],res['_notes']=combine_confidence(res['规则置信度'],res['依赖'],quality)
    if verbose:
        print('='*60); print('              滑雪技术分析报告'); print('='*60)
        print(f'\n【数据质量】总体置信度: {quality["overall"]}')
        for w in quality['warnings']: print(f'  {w}')
        print('\n【检测到的技术要点】')
        trig=[(n,r) for n,r in rules if r['触发']]
        if trig:
            for n,r in trig:
                print(f'\n● {n}  [置信度: {r["_conf"]}]')
                print(f'  {r["建议"]}')
                print(f'  依据: {r["证据"]}')
                if r['_notes']: print(f'  ⚠️ {"; ".join(r["_notes"])}')
        else: print('  未检测到明显技术问题。')
        print(f'\n【正常/数据不足的检查】共{sum(1 for n,r in rules if not r["触发"])}项')
        for n,r in rules:
            if not r['触发']: print(f'  ○ {n} [{r["_conf"]}]: {r["证据"]}')
        print('\n'+'='*60)
        print('注:基于2D姿态估计,受视角影响;阈值为经验值,应结合专业教练意见。')
        print('='*60)
    return rules
