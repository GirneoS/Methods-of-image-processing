"""
Верификация фильтрации (ЛР5).

1) Сохранение энергии (физическая корректность):
    для каждого object_id сумма яркости до и после фильтрации должна совпадать.
2) Подавление шума: сравнение std(noisy) vs std(filtered).
"""

import numpy as np


def luminance(img: np.ndarray) -> np.ndarray:
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def energy_per_object(noisy: np.ndarray, filtered: np.ndarray,
                      obj_id: np.ndarray) -> dict:
    ln, lf = luminance(noisy), luminance(filtered)
    out = {}
    for oid in np.unique(obj_id):
        m = obj_id == oid
        sb = float(ln[m].sum())
        sa = float(lf[m].sum())
        rel = abs(sa - sb) / (sb + 1e-9) * 100
        out[int(oid)] = (sb, sa, rel)
    return out


def noise_stats(noisy: np.ndarray, filtered: np.ndarray) -> dict:
    sn, sf = float(np.std(noisy)), float(np.std(filtered))
    return {
        "std_noisy": sn,
        "std_filtered": sf,
        "reduction_%": (1 - sf / (sn + 1e-9)) * 100,
    }


OBJECT_NAMES = {
    -1: "fon (miss)",
    1:  "pol (floor)",
    2:  "potolok (ceiling)",
    3:  "zadnyaya stena (back)",
    4:  "levaya stena (red)",
    5:  "pravaya stena (green)",
    6:  "korotkij blok (white)",
    7:  "vysokij blok (mirror)",
    8:  "istochnik sveta",
}


def print_report(energy: dict, noise: dict, title: str = "filter") -> None:
    print(f"\n== {title}: energy conservation per object ==")
    print(f"{'id':>3}  {'name':<24} {'sum before':>12}  {'sum after':>12}  {'err %':>7}")
    print("-" * 70)
    tb = ta = 0.0
    for oid, (sb, sa, rel) in sorted(energy.items()):
        name = OBJECT_NAMES.get(oid, f"obj_{oid}")
        print(f"{oid:>3}  {name:<24} {sb:12.2f}  {sa:12.2f}  {rel:6.3f}%")
        tb += sb; ta += sa
    rel_t = abs(ta - tb) / (tb + 1e-9) * 100
    print("-" * 70)
    print(f"{'TOT':>3}  {'TOTAL':<24} {tb:12.2f}  {ta:12.2f}  {rel_t:6.3f}%")

    print(f"\n== {title}: noise reduction ==")
    print(f"  std noisy    : {noise['std_noisy']:.4f}")
    print(f"  std filtered : {noise['std_filtered']:.4f}")
    print(f"  reduction    : {noise['reduction_%']:.2f}%")
