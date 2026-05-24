import numpy as np
import matplotlib.pyplot as plt

from gen import sphere_points

N = 10000

# 1. ТРЕУГОЛЬНИК

def sample_triangle(v1, v2, v3, n):
    r1 = np.random.rand(n)
    r2 = np.random.rand(n)

    mask = (r1 + r2) > 1
    r1[mask] = 1 - r1[mask]
    r2[mask] = 1 - r2[mask]

    points = v1 + np.outer(r1, (v2 - v1)) + np.outer(r2, (v3 - v1))
    return points

v1 = np.array([0, 0])
v2 = np.array([1, 0])
v3 = np.array([0.5, 1])

triangle_points = sample_triangle(v1, v2, v3, N)

# 2. КРУГ

def sample_circle(radius, n):
    r = radius * np.sqrt(np.random.rand(n))
    theta = 2 * np.pi * np.random.rand(n)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return np.column_stack((x, y)), r

circle_points, radii = sample_circle(1.0, N)

# =========================================
# 4. КОСИНУСНОЕ РАСПРЕДЕЛЕНИЕ


# =========================================

def sample_cosine(n):
    r1 = np.random.rand(n)
    r2 = np.random.rand(n)

    theta = np.arccos(np.sqrt(r1))
    phi = 2 * np.pi * r2

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.column_stack((x, y, z)), z

cosine_points, cos_z = sample_cosine(N)

# =========================================
# ВИЗУАЛИЗАЦИЯ (оригинальная)
# =========================================

from matplotlib.patches import Rectangle

# 1. Треугольник
sq1_mask = ((triangle_points[:, 0] >= 0.4) & (triangle_points[:, 0] <= 0.6) &
            (triangle_points[:, 1] >= 0.0) & (triangle_points[:, 1] <= 0.2))
sq2_mask = ((triangle_points[:, 0] >= 0.4) & (triangle_points[:, 0] <= 0.6) &
            (triangle_points[:, 1] >= 0.4) & (triangle_points[:, 1] <= 0.6))
sq3_mask = ((triangle_points[:, 0] >= 0.4) & (triangle_points[:, 0] <= 0.6) &
            (triangle_points[:, 1] >= 0.6) & (triangle_points[:, 1] <= 0.8))
count_tri1 = int(np.sum(sq1_mask))
count_tri2 = int(np.sum(sq2_mask))
count_tri3 = int(np.sum(sq3_mask))

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(triangle_points[:, 0], triangle_points[:, 1], s=1)
for (xmin, xmax, ymin, ymax), color, label in [
    ((0.4, 0.6, 0.0, 0.2), 'red',    f'Квадрат 1: {count_tri1}'),
    ((0.4, 0.6, 0.4, 0.6), 'green',  f'Квадрат 2: {count_tri2}'),
    ((0.4, 0.6, 0.6, 0.8), 'orange', f'Квадрат 3: {count_tri3}'),
]:
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                            linewidth=2, edgecolor=color, facecolor=color,
                            alpha=0.35, label=label))
ax.set_title("Треугольник: равномерное распределение")
ax.set_aspect('equal')
ax.legend()
plt.show()

# 2. Круг
s = 1 / np.sqrt(2)  # 0.707 — сторона квадрата вписанного в единичный круг
regions_circle = [
    (-s,  0, -s,  0),
    ( 0,  s, -s,  0),
    (-s,  0,  0,  s),
    ( 0,  s,  0,  s),
]
counts_circle = []
for xmin, xmax, ymin, ymax in regions_circle:
    mask = ((circle_points[:, 0] >= xmin) & (circle_points[:, 0] <= xmax) &
            (circle_points[:, 1] >= ymin) & (circle_points[:, 1] <= ymax))
    counts_circle.append(int(np.sum(mask)))

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(circle_points[:, 0], circle_points[:, 1], s=1)
for (xmin, xmax, ymin, ymax), color, count in zip(
        regions_circle, ['red', 'green', 'orange', 'purple'], counts_circle):
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                            linewidth=2, edgecolor=color, facecolor='none',
                            alpha=1.0, label=f'Квадрат: {count}'))
ax.set_title("Круг: равномерное распределение")
ax.set_aspect('equal')
ax.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2], s=1)
ax.set_title("Сфера: равномерное распределение")
ax.set_box_aspect([1,1,1])
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cosine_points[:, 0], cosine_points[:, 1], cosine_points[:, 2], s=1)
ax.set_title("Косинусное распределение (вдоль Z)")
ax.set_box_aspect([1,1,1])
plt.show()

# =========================================
# ДОКАЗАТЕЛЬСТВО РАВНОМЕРНОСТИ / НЕРАВНОМЕРНОСТИ
# =========================================

print("=== ДОКАЗАТЕЛЬСТВО РАВНОМЕРНОСТИ ===\n")

print(f"Треугольник - 3 квадрата:")
print(f"   Квадрат 1: {count_tri1} точек")
print(f"   Квадрат 2: {count_tri2} точек")
print(f"   Квадрат 3: {count_tri3} точек")
print(f"   Среднее: {np.mean([count_tri1, count_tri2, count_tri3]):.0f} \n")

plt.figure(figsize=(8, 4))
plt.bar(['Квадрат 1', 'Квадрат 2', 'Квадрат 3'],
        [count_tri1, count_tri2, count_tri3])
plt.title("Треугольник: кол-во точек в 3 равновеликих квадратах\n")
plt.ylabel("Количество точек")
plt.show()

print(f"Круг — 4 квадрата:")
for i, c in enumerate(counts_circle):
    print(f"   Квадрат {i+1}: {c} точек")
print(f"   Среднее: {np.mean(counts_circle):.0f}\n")

print("=== ДОКАЗАТЕЛЬСТВО РАВНОМЕРНОСТИ ДЛЯ СФЕРЫ ===\n")

np.random.seed(42)

random_directions = np.random.randn(3, 3)
random_directions /= np.linalg.norm(random_directions, axis=1)[:, np.newaxis]

alpha = np.deg2rad(25)
cos_alpha = np.cos(alpha)

counts_cap = []
cap_points = []

#print(f"Угол: {np.rad2deg(alpha):.1f}°, cos(alpha) = {cos_alpha:.4f}\n")

for i, axis in enumerate(random_directions):
    # Скалярное произведение всех точек сферы с текущей осью
    dots = np.dot(sphere_points, axis)

    # Точки внутри колпака: dot >= cos(alpha)
    mask = dots >= cos_alpha
    count = np.sum(mask)
    counts_cap.append(count)

    # Сохраняем точки для визуализации
    cap_points.append(sphere_points[mask])

    print(f"Колпак {i + 1}: "
          f"{count} точек")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2],
           s=1, alpha=0.1, color='gray', label='Все точки')

colors = ['red', 'green', 'blue']
for i, pts in enumerate(cap_points):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               s=8, color=colors[i], label=f'Колпак {i + 1}')

    ax.quiver(0, 0, 0,
              random_directions[i, 0] * 1,
              random_directions[i, 1] * 1,
              random_directions[i, 2] * 1,
              color=colors[i], linewidth=3)

ax.set_title("Сфера: \n")
ax.set_box_aspect([1, 1, 1])
ax.legend()
plt.show()

np.random.seed(42)  # для воспроизводимости

# === Параметры конуса ===
alpha_deg = 30
alpha = np.deg2rad(alpha_deg)
cos_alpha = np.cos(alpha)

#print(f"Угол конуса: {alpha_deg}°,  cos(α) = {cos_alpha:.4f}")

# Генерируем 6 случайных направлений
directions = np.random.randn(6, 3)
directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]

# Таблица для косинусного распределения
# Таблица для косинусного распределения
print("Косинусное распределение:")
print("Напр. | Вектор направления (x, y, z) | cos к +Z | Угол к +Z | Точек в конусе")
print("-" * 85)

results = []
for i, axis in enumerate(directions):
    # Нормируем направление на всякий случай
    axis_norm = axis / np.linalg.norm(axis)

    # Для косинусного распределения нужно отразить точки,
    # если направление смотрит вниз
    if axis_norm[2] < 0:
        # Отражаем все точки относительно плоскости XY
        # или поворачиваем направление на 180° вокруг X или Y
        adjusted_axis = -axis_norm
    else:
        adjusted_axis = axis_norm

    dots = np.dot(cosine_points, adjusted_axis)
    count = np.sum(dots >= cos_alpha)

    cos_to_z = adjusted_axis[2]
    angle_to_z = np.rad2deg(np.arccos(np.clip(cos_to_z, -1.0, 1.0)))

    results.append({
        'index': i + 1,
        'axis': axis,
        'cos_to_z': cos_to_z,
        'angle_to_z': angle_to_z,
        'count': count
    })

# Сортируем по количеству точек в конусе (по убыванию)
sorted_results = sorted(results, key=lambda x: x['count'], reverse=True)

# Выводим отсортированные результаты
for item in sorted_results:
    print(
        f"{item['index']:2d}    | [{item['axis'][0]:6.3f}, {item['axis'][1]:6.3f}, {item['axis'][2]:6.3f}] | "
        f"{item['cos_to_z']:8.4f} | {item['angle_to_z']:6.1f}°  | {item['count']:6d}"
    )
