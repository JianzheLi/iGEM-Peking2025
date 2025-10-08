import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from tqdm import tqdm
# ------------------------------
# 胃边界曲线
# ------------------------------
def x_small(t):
    return 15*(-67.7032*t**8 + 214.1229*t**7 - 234.5024*t**6 + 75.7629*t**5 +
               46.5178*t**4 - 46.9845*t**3 + 12.7277*t**2 - 0.2367*t + 0.2974)

def y_small(t):
    return 15*(-477.5217*t**8 + 1921.5726*t**7 - 3097.9677*t**6 + 2546.8035*t**5 -
               1125.1176*t**4 + 257.9309*t**3 - 25.4794*t**2 - 1.0590*t - 0.0332)

def x_large(t):
    return 15*(226.1847*t**8 - 949.5591*t**7 + 1565.2398*t**6 - 1271.1764*t**5 +
               530.0613*t**4 - 116.1492*t**3 + 14.9300*t**2 + 0.2104*t + 0.3495)

def y_large(t):
    return 15*(227.3934*t**8 - 621.4740*t**7 + 440.2322*t**6 + 161.8485*t**5 -
               300.7059*t**4 + 96.3826*t**3 - 1.8308*t**2 - 2.7398*t - 0.0010)

t_vals = np.linspace(0, 1, 1000)
xs, ys = x_small(t_vals), y_small(t_vals)
xl, yl = x_large(t_vals), y_large(t_vals)

# ------------------------------
# 胃模型
# ------------------------------
class Stomach:
    def __init__(self, wall_thickness=0.5,inlet_frac=0.05, outlet_frac=0.05):
        self.t_vals = t_vals
        self.x_small_vals = xs
        self.y_small_vals = ys
        self.x_large_vals = xl
        self.y_large_vals = yl
        self.wall_thickness = wall_thickness

        # 胃纵向范围
        self.ymin, self.ymax = min(ys.min(), yl.min()), max(ys.max(), yl.max())


        # 胃边界多边形
        from matplotlib.path import Path
        poly_x = np.concatenate([self.x_small_vals, self.x_large_vals[::-1]])
        poly_y = np.concatenate([self.y_small_vals, self.y_large_vals[::-1]])
        self.boundary_polygon = Path(np.column_stack((poly_x, poly_y)))

        # 入口出口区域
        self.ymin, self.ymax = min(ys.min(), yl.min()), max(ys.max(), yl.max())
        # 设置入口和出口属性
        self.inlet_t_range = (0.15 , 0.15+inlet_frac)   # t 接近 1
        self.outlet_t_range = (1.0-outlet_frac, 1.0)     # t 接近 0

        # 用于生成微球
        self.width = max(xl.max(), xs.max()) + wall_thickness
        self.height = self.ymax + wall_thickness
    def _t_for_point(self, x, y, curve='small'):
        """找到点在曲线上对应的 t 值"""
        vals = self.x_small_vals if curve=='small' else self.x_large_vals
        dists = np.sqrt((vals - x)**2 + (self.y_small_vals - y)**2) if curve=='small' else np.sqrt((vals - x)**2 + (self.y_large_vals - y)**2)
        idx = np.argmin(dists)
        return self.t_vals[idx]
    def is_inside(self, x, y):
        
        return self.boundary_polygon.contains_point((x, y))

        if y < self.ymin or y > self.ymax:
            return False
        idx_small = np.argmin(np.abs(self.y_small_vals - y))
        idx_large = np.argmin(np.abs(self.y_large_vals - y))
        x_left = self.x_small_vals[idx_small]
        x_right = self.x_large_vals[idx_large]
        return x_left <= x <= x_right
    def get_flow_direction(self, x, y):
        point = np.array([x, y])

        # 小弯最近点
        small_points = np.column_stack((self.x_small_vals, self.y_small_vals))
        dists_small = np.linalg.norm(small_points - point, axis=1)
        idx_small = np.argmin(dists_small)
        t_small = self.t_vals[idx_small]

        # 大弯最近点
        large_points = np.column_stack((self.x_large_vals, self.y_large_vals))
        dists_large = np.linalg.norm(large_points - point, axis=1)
        idx_large = np.argmin(dists_large)
        t_large = self.t_vals[idx_large]

        # 速度场（dx/dt, dy/dt）
        dxs = np.gradient(self.x_small_vals, self.t_vals)
        dys = np.gradient(self.y_small_vals, self.t_vals)
        dxl = np.gradient(self.x_large_vals, self.t_vals)
        dyl = np.gradient(self.y_large_vals, self.t_vals)

        v_small = np.array([dxs[idx_small], dys[idx_small]])
        v_large = np.array([dxl[idx_large], dyl[idx_large]])

        # 权重（依旧用胃的分段位置决定）
        frac = 0.5 * (t_small + t_large) / self.t_vals[-1]
        if frac < 0.4:
            w_small = 0.5
        elif frac < 0.9:
            w_small = 0.3
        else:
            w_small = 0.3
        w_large = 1.0 - w_small

        # 合成流速
        v = w_small * v_small + w_large * v_large
        return v / (np.linalg.norm(v) + 1e-8)

    def get_flow_direction1(self, x, y):
        idx = np.argmin(np.abs(self.y_small_vals - y))
        dxs = np.gradient(self.x_small_vals, self.t_vals)
        dys = np.gradient(self.y_small_vals, self.t_vals)
        dxl = np.gradient(self.x_large_vals, self.t_vals)
        dyl = np.gradient(self.y_large_vals, self.t_vals)
        v_small = np.array([dxs[idx], dys[idx]])
        v_large = np.array([dxl[idx], dyl[idx]])
        total_points = len(self.t_vals)
        if idx < total_points*0.3:
            w_small = 0.7
        elif idx < total_points*(0.9):
            w_small = 0.3
        else:
            w_small = 0.3
        w_large = 1.0 - w_small

        v = w_small * v_small + w_large * v_large
        return v / (np.linalg.norm(v) + 1e-8)

    def is_in_inlet_area(self, x, y):
        t_small = self._t_for_point(x, y, 'small')
        t_large = self._t_for_point(x, y, 'large')
        # 如果两个 t 都在 inlet 范围内，则认为在入口
        return (self.inlet_t_range[0] <= t_small <= self.inlet_t_range[1] and
                self.inlet_t_range[0] <= t_large <= self.inlet_t_range[1])

    def is_in_outlet_area(self, x, y):
        t_small = self._t_for_point(x, y, 'small')
        t_large = self._t_for_point(x, y, 'large')
        return (self.outlet_t_range[0] <= t_small <= self.outlet_t_range[1] and
                self.outlet_t_range[0] <= t_large <= self.outlet_t_range[1])
    def is_on_wall1(self, x, y, radius=0.1):
        
        if self.is_in_inlet_area(x, y) or self.is_in_outlet_area(x, y):
            return False
        points = np.vstack([np.column_stack((self.x_small_vals, self.y_small_vals)),
                            np.column_stack((self.x_large_vals, self.y_large_vals))])
        min_dist = np.min(cdist([[x, y]], points))
        return min_dist <= radius
    def _point_to_segment_dist(self, px, py, x1, y1, x2, y2):
        v = np.array([x2 - x1, y2 - y1])
        w = np.array([px - x1, py - y1])
        c1 = np.dot(w, v)
        if c1 <= 0:
            return np.linalg.norm(w)
        c2 = np.dot(v, v)
        if c2 <= c1:
            return np.linalg.norm([px - x2, py - y2])
        b = c1 / c2
        pb = np.array([x1, y1]) + b * v
        return np.linalg.norm([px - pb[0], py - pb[1]])
    def is_on_wall(self, x, y, radius=0.1):
        # 出口和入口区域不算胃壁
        if self.is_in_inlet_area(x, y) or self.is_in_outlet_area(x, y):
            return False
        
        # 遍历胃边界的所有线段，计算最近距离
        vertices = self.boundary_polygon.vertices
        min_dist = float("inf")
        for i in range(len(vertices)):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i+1) % len(vertices)]  # 下一个点，首尾相连
            dist = self._point_to_segment_dist(x, y, x1, y1, x2, y2)
            if dist < min_dist:
                min_dist = dist
        
        return min_dist <= radius
# ------------------------------
# 微球模型
# ------------------------------
class Microsphere:
    def __init__(self, stomach, half_life, x, y, radius=0.1, p_attach=1):
        self.stomach = stomach
        self.radius = radius
        self.half_life = half_life
        self.x, self.y = x, y
        self.status = 0  # 0-free, 1-attached, 2-exited
        self.diffusion_coefficient = 0.1
        self.liquid_flow_speed = 0.5
        self.decay_constant = np.log(2)/half_life
        self.activity = 1.0
        self.p_attach = p_attach
        self.vx, self.vy = 0.0, 0.0

    def update(self, dt):
        if self.status != 0:
            self.activity *= np.exp(-self.decay_constant*dt)
            return

        # 活性衰减
        self.activity *= np.exp(-self.decay_constant*dt)

        # 扩散
        dx_diff = np.random.normal(0, np.sqrt(2*self.diffusion_coefficient*dt))
        dy_diff = np.random.normal(0, np.sqrt(2*self.diffusion_coefficient*dt))

        # 流动
        flow_dir = self.stomach.get_flow_direction(self.x, self.y)
        self.vx = 0.5*flow_dir[0]*self.liquid_flow_speed+0.5*self.vx
        self.vy = 0.5*flow_dir[1]*self.liquid_flow_speed+0.5*self.vy
        flow_dir = np.array([self.vx, self.vy])
        flow_dir /= (np.linalg.norm(flow_dir)+1e-8)
        flow_dir[1]-=0.01
        dx_flow, dy_flow = flow_dir*self.liquid_flow_speed*dt

        new_x, new_y = self.x + dx_diff + dx_flow, self.y + dy_diff + dy_flow

        # 出界 → 排出
        if not self.stomach.is_inside(new_x, new_y):
            self.status = 0.0
            return

        # 胃壁附着或反弹
        if self.stomach.is_on_wall(new_x, new_y, self.radius):
            if np.random.rand() < self.p_attach:
                self.status = 1
                self.x, self.y = new_x, new_y
                return
            else:
                normal = self._wall_normal(new_x, new_y)
                v = np.array([dx_diff + dx_flow, dy_diff + dy_flow])
                v_reflect = v - 2*np.dot(v, normal)*normal
                new_x, new_y = self.x + v_reflect[0], self.y + v_reflect[1]

        self.x, self.y = new_x, new_y

    def _wall_normal(self, x, y):
        points = np.vstack([np.column_stack((self.stomach.x_small_vals, self.stomach.y_small_vals)),
                            np.column_stack((self.stomach.x_large_vals, self.stomach.y_large_vals))])
        nearest_idx = np.argmin(np.linalg.norm(points - np.array([x, y]), axis=1))
        # 局部切向量
        if nearest_idx < len(self.stomach.x_small_vals)-1:
            tx = self.stomach.x_small_vals[nearest_idx+1]-self.stomach.x_small_vals[nearest_idx]
            ty = self.stomach.y_small_vals[nearest_idx+1]-self.stomach.y_small_vals[nearest_idx]
        else:
            idx = nearest_idx - len(self.stomach.x_small_vals)
            tx = self.stomach.x_large_vals[idx+1]-self.stomach.x_large_vals[idx]
            ty = self.stomach.y_large_vals[idx+1]-self.stomach.y_large_vals[idx]
        tangent = np.array([tx, ty])
        tangent /= (np.linalg.norm(tangent)+1e-8)
        normal = np.array([-tangent[1], tangent[0]])
        normal /= (np.linalg.norm(normal)+1e-8)
        return normal

# ------------------------------
# 模拟管理器
# ------------------------------
class Simulation:
    def __init__(self, num_microspheres=1000, simulation_time=600, half_life=300, dt=1,snapshot_interval=60):
        self.stomach = Stomach()
        self.total_spheres = num_microspheres
        self.sim_time = simulation_time
        self.half_life = half_life
        self.dt = dt
        self.snapshot_interval = snapshot_interval
        self.current_time = 0
        self.last_snapshot_time=0
        self.microspheres = []
        self.spheres_generated = 0

        # Statistics
        self.time_points = []
        self.free_count = []
        self.attached_count = []
        self.exited_count = []

    def step(self):
        if self.current_time >= self.sim_time:
            return
        self._generate_microspheres()
        for s in self.microspheres:
            s.update(self.dt)
        self._record_stats()
        self.current_time += self.dt
        
        if self.current_time - self.last_snapshot_time >= self.snapshot_interval:
            self.save_snapshot()
            self.last_snapshot_time = self.current_time

        self.current_time += self.dt


    def _generate_microspheres(self):
        """Generate microspheres along the inlet curve based on t range"""
        remaining = self.total_spheres - self.spheres_generated
        if remaining <= 0:
            return
        per_step = max(1, remaining // (self.sim_time - self.current_time + 1))
        num_generate = remaining

        t_min, t_max = self.stomach.inlet_t_range
        for _ in range(num_generate):
            # 在 t_min~t_max 范围随机生成 t
            t = np.random.uniform(t_min, t_max)
            # 在小弯和大弯之间随机插值生成 x
            x_small_val = np.interp(t, self.stomach.t_vals, self.stomach.x_small_vals)
            x_large_val = np.interp(t, self.stomach.t_vals, self.stomach.x_large_vals)
            y_val = np.interp(t, self.stomach.t_vals, self.stomach.y_small_vals)  # y 小弯/大弯差不多
            x = np.random.uniform(x_small_val, x_large_val)
            y = y_val

            self.microspheres.append(Microsphere(self.stomach, self.half_life, x, y))
            self.spheres_generated += 1

    def _record_stats(self):
        free = sum(1 for s in self.microspheres if s.status == 0)
        attached = sum(1 for s in self.microspheres if s.status == 1)
        exited = sum(1 for s in self.microspheres if s.status == 2)
        self.time_points.append(self.current_time)
        self.free_count.append(free)
        self.attached_count.append(attached)
        self.exited_count.append(exited)

    def run(self):
        while self.current_time < self.sim_time:
            self.step()

    def visualize(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # 胃曲线
        ax1.plot(self.stomach.x_small_vals, self.stomach.y_small_vals, 'k')
        ax1.plot(self.stomach.x_large_vals, self.stomach.y_large_vals, 'k')

        # 微球状态
        colors = {0: 'blue', 1: 'red', 2: 'purple'}
        labels = {0: 'Free', 1: 'Attached', 2: 'Exited'}
        for status in [0, 1, 2]:
            x = [s.x for s in self.microspheres if s.status == status]
            y = [s.y for s in self.microspheres if s.status == status]
            ax1.scatter(x, y, s=5, color=colors[status], label=labels[status], alpha=0.6)

        ax1.set_title(f'Microsphere Distribution at t={self.sim_time}s')
        ax1.set_xlabel('X (cm)')
        ax1.set_ylabel('Y (cm)')
        ax1.legend()
        ax1.set_aspect('equal')

        # 时间序列
        #ax2.plot(self.time_points, self.free_count, label='Free', color='blue')
        ax2.plot(self.time_points, self.attached_count, label='Attached', color='red')
        ax2.plot(self.time_points, self.exited_count, label='Exited', color='purple')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Count')
        ax2.set_title('Microsphere Status Over Time')
        ax2.legend()
        plt.savefig("microsphere_simulation.png")
        plt.show()
        
    def save_snapshot(self):
        """保存当前微球分布图片"""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.stomach.x_small_vals, self.stomach.y_small_vals, 'k')
        ax.plot(self.stomach.x_large_vals, self.stomach.y_large_vals, 'k')
        colors = {0: 'blue', 1: 'red', 2: 'purple'}
        labels = {0: 'Free', 1: 'Attached', 2: 'Exited'}
        for status in [0, 1, 2]:
            x = [s.x for s in self.microspheres if s.status == status]
            y = [s.y for s in self.microspheres if s.status == status]
            ax.scatter(x, y, s=5, color=colors[status], label=labels[status], alpha=0.6)
        ax.set_aspect('equal')
        ax.set_title(f'Microsphere Distribution at t={self.current_time}s')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.legend()
        plt.tight_layout()
        filename = f"snapshot_{int(self.current_time)}s.png"
        plt.savefig(filename)
        plt.close(fig)  # 关闭 figure 释放内存
        print(f"Saved snapshot: {filename}")
        
    def visualize_flow_field(self, grid_size=100):
        """
        可视化胃内部流场，grid_size 控制采样密度
        """
        # 在胃内部生成网格点
        x_min, x_max = min(self.stomach.x_small_vals.min(), self.stomach.x_large_vals.min()), \
                    max(self.stomach.x_small_vals.max(), self.stomach.x_large_vals.max())
        y_min, y_max = self.stomach.ymin, self.stomach.ymax

        xs = np.linspace(x_min, x_max, grid_size)
        ys = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(xs, ys)
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        speed = np.zeros_like(X)

        for i in range(grid_size):
            for j in range(grid_size):
                x, y = X[i, j], Y[i, j]
                if self.stomach.is_inside(x, y):
                    v = self.stomach.get_flow_direction(x, y)
                    U[i, j], V[i, j] = v
                    speed[i, j] = np.linalg.norm(v)
                else:
                    U[i, j], V[i, j] = 0, 0
                    speed[i, j] = 0

        # 绘图
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.stomach.x_small_vals, self.stomach.y_small_vals, 'k')
        ax.plot(self.stomach.x_large_vals, self.stomach.y_large_vals, 'k')
        # 使用 quiver 绘制箭头，颜色表示速度大小
        q = ax.quiver(X, Y, U, V, speed, scale=30, cmap='jet', width=0.003)
        cbar = plt.colorbar(q, ax=ax)
        cbar.set_label('Flow speed')
        ax.set_aspect('equal')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_title('Stomach Flow Field')
        plt.savefig("flow_field.png")
        plt.show()

# ------------------------------
# 主程序
# ------------------------------
if __name__=="__main__":
    """num = int(input("Enter number of microspheres: "))
    sim_time = int(input("Enter simulation time (s): "))
    half_life = int(input("Enter microsphere half-life (s): "))"""
    num = 1000
    sim_time = 600
    half_life =500
    sim = Simulation(num, sim_time, half_life)
    sim.visualize_flow_field(grid_size=25)
    sim.run()
    sim.visualize()
    # 在 Stomach 类初始化后（比如 sim = Simulation(...) 之后）
    stomach = sim.stomach

    # 取出多边形顶点
    verts = stomach.boundary_polygon.vertices

    # 绘制胃的边界多边形
    """import matplotlib.pyplot as plt
    plt.figure(figsize=(6,8))
    plt.plot(verts[:,0], verts[:,1], 'r-', lw=2, label="Boundary polygon")
    plt.plot(stomach.x_small_vals, stomach.y_small_vals, 'b--', label="Small curve")
    plt.plot(stomach.x_large_vals, stomach.y_large_vals, 'g--', label="Large curve")
    plt.gca().set_aspect("equal")
    plt.legend()
    plt.title("Check stomach boundary polygon")
    plt.show()"""
