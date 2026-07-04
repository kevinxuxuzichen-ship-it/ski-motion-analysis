"""
ski_physics.py — 双板滑雪姿态物理量计算
输入:24个关键点 [x, y, v]  (x,y归一化或像素均可;v: 1=可见, 0=遮挡)
作者备注:所有"绝对角度"假设相机大致端平;2D量受透视影响,详见各函数说明。
"""
import numpy as np

# ========== 24点索引 ==========
HEAD, NECK = 0, 1
SH_R, EL_R, WR_R = 2, 3, 4          # 右肩/肘/腕
PB_R = 5                             # 右杖篮
SH_L, EL_L, WR_L = 6, 7, 8          # 左肩/肘/腕
PB_L = 9                             # 左杖篮
HIP_R, KNEE_R, ANK_R = 10, 11, 12   # 右髋/膝/踝
HIP_L, KNEE_L, ANK_L = 13, 14, 15   # 左髋/膝/踝
SKI_TIP_R, TOE_R, HEEL_R, SKI_TAIL_R = 16, 17, 18, 19   # 右板尖/脚尖/脚跟/板尾
SKI_TIP_L, TOE_L, HEEL_L, SKI_TAIL_L = 20, 21, 22, 23   # 左板

# ========== 工具 ==========
def midpoint(pa, pb):
    """两点中点,可见性取两者最小"""
    pa, pb = np.asarray(pa, float), np.asarray(pb, float)
    return np.array([(pa[0]+pb[0])/2, (pa[1]+pb[1])/2, min(pa[2], pb[2])])


# ========== A: 关节夹角 ==========
def joint_angle(p1, p2, p3, vis_thresh=1, eps=1e-8):
    """顶点在p2的夹角(0~180°)。返回(角度, 可信)。三层保护:可见性/防除零/clip。"""
    p1, p2, p3 = np.asarray(p1,float), np.asarray(p2,float), np.asarray(p3,float)
    trustworthy = (p1[2]>=vis_thresh and p2[2]>=vis_thresh and p3[2]>=vis_thresh)
    a = p1[:2] - p2[:2]
    b = p3[:2] - p2[:2]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.nan, False
    cos_theta = np.clip(np.dot(a, b)/(na*nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta))), trustworthy


def all_joint_angles(ann, vis_thresh=1):
    """一次算全身关节角。返回 {名称: (角度, 可信)}。"""
    defs = [
        ('右膝', HIP_R, KNEE_R, ANK_R), ('左膝', HIP_L, KNEE_L, ANK_L),
        ('右髋', SH_R,  HIP_R,  KNEE_R),('左髋', SH_L,  HIP_L,  KNEE_L),
        ('右肘', SH_R,  EL_R,   WR_R),  ('左肘', SH_L,  EL_L,   WR_L),
    ]
    return {name: joint_angle(ann[i1], ann[i2], ann[i3], vis_thresh)
            for name, i1, i2, i3 in defs}


# ========== C: 身体线倾斜 & 反弓 ==========
def body_line_tilt(p_low, p_high, vis_thresh=1, eps=1e-8):
    """身体线相对图像垂直方向的带符号倾斜角。0°=竖直, +上端偏右, -上端偏左。
       注意:y轴朝下;'相对重力'仅在相机端平时成立。返回(角度, 可信)。"""
    p_low, p_high = np.asarray(p_low,float), np.asarray(p_high,float)
    trustworthy = (p_low[2]>=vis_thresh and p_high[2]>=vis_thresh)
    vx = p_high[0] - p_low[0]
    vy = p_high[1] - p_low[1]
    if (vx*vx + vy*vy)**0.5 < eps:
        return np.nan, False
    return float(np.degrees(np.arctan2(vx, -vy))), trustworthy


def angulation(p_hip, p_knee, p_sh_mid, p_hip_mid, vis_thresh=1):
    """反弓角 = 大腿内倾 - 躯干内倾。对相机倾斜/坡度/前后镜像鲁棒。返回(角度, 可信)。"""
    thigh, ok_t = body_line_tilt(p_knee, p_hip, vis_thresh)
    trunk, ok_r = body_line_tilt(p_hip_mid, p_sh_mid, vis_thresh)
    if np.isnan(thigh) or np.isnan(trunk):
        return np.nan, False
    return float(thigh - trunk), (ok_t and ok_r)


# ========== B: 重心 ==========
# (名称, 端点A, 端点B, 质量比例);trunk特殊处理。系数来自Dempster简化值。
_SEGMENTS = [
    ('head', HEAD, NECK, 0.08), ('trunk', None, None, 0.50),
    ('uarm_R', SH_R, EL_R, 0.027), ('uarm_L', SH_L, EL_L, 0.027),
    ('farm_R', EL_R, WR_R, 0.022), ('farm_L', EL_L, WR_L, 0.022),
    ('thigh_R', HIP_R, KNEE_R, 0.10), ('thigh_L', HIP_L, KNEE_L, 0.10),
    ('shank_R', KNEE_R, ANK_R, 0.06), ('shank_L', KNEE_L, ANK_L, 0.06),
]

def center_of_mass(ann, vis_thresh=1, min_mass=0.70):
    """分段质量模型算2D重心。返回 (com_x, com_y, 覆盖质量, 可信)。
       缺失段跳过并重新归一化;覆盖质量<min_mass则不可信。"""
    tot, wx, wy = 0.0, 0.0, 0.0
    for name, ia, ib, mass in _SEGMENTS:
        if name == 'trunk':
            pa = midpoint(ann[SH_R], ann[SH_L])
            pb = midpoint(ann[HIP_R], ann[HIP_L])
        else:
            pa, pb = np.asarray(ann[ia],float), np.asarray(ann[ib],float)
        if pa[2] < vis_thresh or pb[2] < vis_thresh:
            continue
        wx += mass * (pa[0]+pb[0])/2
        wy += mass * (pa[1]+pb[1])/2
        tot += mass
    if tot < 1e-6:
        return np.nan, np.nan, 0.0, False
    return wx/tot, wy/tot, tot, (tot >= min_mass)


# ========== 一键提取所有物理量 ==========
def extract_all(ann, vis_thresh=1):
    """对一帧的24点,算出所有物理量,打包成dict。"""
    sh_mid  = midpoint(ann[SH_R], ann[SH_L])
    hip_mid = midpoint(ann[HIP_R], ann[HIP_L])
    out = {}
    out['joints'] = all_joint_angles(ann, vis_thresh)
    out['trunk_tilt'] = body_line_tilt(hip_mid, sh_mid, vis_thresh)
    out['angulation_R'] = angulation(ann[HIP_R], ann[KNEE_R], sh_mid, hip_mid, vis_thresh)
    out['angulation_L'] = angulation(ann[HIP_L], ann[KNEE_L], sh_mid, hip_mid, vis_thresh)
    cx, cy, mcov, ok = center_of_mass(ann, vis_thresh)
    out['com'] = (cx, cy, mcov, ok)
    return out
