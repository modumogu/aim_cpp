from pathlib import Path
BLOCK = """

# ---------------------------------------------------------------------------
# 亮度 / 饱和度自适应（压力测试后新增，正常光照下可以不改）
# ---------------------------------------------------------------------------
# 画面整体偏暗时自动补增益（软件自动曝光），避免 V 阈值把灯条一起挡掉。
# 只有 ROI 内亮度的 P99 低于 adapt_gain_below 才补，正常光照不会触发。
adapt_sv = 1
adapt_gain_below = 110
adapt_max_gain = 4.0

# 饱和度自适应：只有"画面里最亮的那些像素"本身就偏淡（S<120）时才放宽 S 下限，
# 用于应对褪色/白平衡失配。暖光下灯条仍然饱和，不会触发，避免背景混入。
adapt_ratio = 0.45
adapt_min = 45
"""
for name in ("config.txt", "config_robomini.txt"):
    p = Path(r"D:\学习\aim_cpp") / name
    t = p.read_text(encoding="utf-8")
    if "adapt_gain_below" in t:
        print(f"  [已有] {name}")
        continue
    p.write_text(t.rstrip() + BLOCK, encoding="utf-8", newline="\n")
    ok = "adapt_gain_below" in p.read_text(encoding="utf-8")
    print(f"  [{('成功' if ok else '失败')}] {name}")
    # 同步到工作区
    q = Path(r"D:\学习\机甲杯\aim_cpp") / name
    q.write_text(p.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
