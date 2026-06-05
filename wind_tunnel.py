#!/usr/bin/env python3
# wind_tunnel.py  -- desktop port (pygame)
# Wind tunnel simulator with real Navier-Stokes physics + rigid-body tipping.
#
# Original: Pythonista 3 (iOS) scene+ui by vvas2002.
# This file is a standalone desktop port. Core physics (FluidGrid, RigidBody,
# Shape, particles, force integration) is IDENTICAL to the iOS version.
# Only the rendering / UI / input layer was rewritten:
#     scene.Scene  -> pygame main loop
#     ui.Slider/Switch/Button/TextField/SegmentedControl -> hand-drawn widgets
#     photos.pick_asset() -> tkinter file dialog
#
# IMPORTANT COORDINATE NOTE:
#   Pythonista scene: origin bottom-left, "down" = -y.
#   pygame:           origin top-left,    "down" = +y.
#   All gravity / ground / rendering was flipped accordingly. The fluid grid
#   and shape math are coordinate-agnostic so they were left untouched.

import math
import random
import sys
import os

import pygame

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

# ---------- config ----------
MAX_PARTICLES = 450
TRAIL_LEN = 5
PARTICLE_LIFETIME = 8.0
SPAWN_BASE = 70
FLAT_THRESHOLD_DEG = 15
BAD_MARKER_LIFE = 2.5

FX, FY = 60, 34
HEAT_W, HEAT_H = FX, FY

NS_ITERS = 10
DIFFUSE_ITERS = 2

# ---------- rigid body physics config ----------
GRAVITY = 900.0
GROUND_MARGIN = 40       # px from the bottom edge of the canvas
BODY_DENSITY = 0.00018
REST_DAMP = 0.78
ANG_DAMP = 0.86
SETTLE_EPS = 6.0
RIP_MARGIN = 0.3
TORN_SPIN = 4.0
GROUND_FRICTION = 0.55

SIDEBAR_W = 240

# ---------- math helpers ----------
def v_add(a, b):  return (a[0]+b[0], a[1]+b[1])
def v_sub(a, b):  return (a[0]-b[0], a[1]-b[1])
def v_mul(a, k):  return (a[0]*k, a[1]*k)
def v_dot(a, b):  return a[0]*b[0] + a[1]*b[1]
def v_len(a):     return math.hypot(a[0], a[1])
def v_norm(a):
    L = v_len(a)
    return (a[0]/L, a[1]/L) if L > 1e-9 else (0.0, 0.0)

def seg_normal(p1, p2):
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    n = (-dy, dx)
    return v_norm(n)

def seg_seg_intersect(p1, p2, p3, p4):
    x1,y1=p1; x2,y2=p2; x3,y3=p3; x4,y4=p4
    den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(den) < 1e-9: return None
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / den
    u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / den
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return (x1 + t*(x2-x1), y1 + t*(y2-y1)), t
    return None

def point_in_polygon(pt, pts):
    x, y = pt
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]; xj, yj = pts[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside

# ---------- shape ----------
class Shape:
    def __init__(self, pts):
        if len(pts) < 3:
            self.valid = False
            self.segments = []
            self.pts = pts
            self.centroid = (0,0)
            return
        if pts[0] != pts[-1]:
            pts = pts + [pts[0]]
        self.pts = pts
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        self.centroid = (cx, cy)
        self.segments = []
        for i in range(len(pts)-1):
            a, b = pts[i], pts[i+1]
            n = seg_normal(a, b)
            mid = ((a[0]+b[0])*0.5, (a[1]+b[1])*0.5)
            to_center = v_sub(self.centroid, mid)
            if v_dot(n, to_center) > 0:
                n = (-n[0], -n[1])
            self.segments.append((a, b, n))
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        self.bbox = (min(xs), min(ys), max(xs), max(ys))
        self.valid = True
        area = 0.0
        for i in range(len(pts)-1):
            x0, y0 = pts[i]
            x1, y1 = pts[i+1]
            area += x0*y1 - x1*y0
        self.area = abs(area) * 0.5
        self.char_length = math.sqrt(max(self.area, 1.0))
        self.height = max(ys) - min(ys)
        self.width  = max(xs) - min(xs)

    def collide(self, p_prev, p_now):
        minx, miny, maxx, maxy = self.bbox
        if max(p_prev[0], p_now[0]) < minx or min(p_prev[0], p_now[0]) > maxx: return None
        if max(p_prev[1], p_now[1]) < miny or min(p_prev[1], p_now[1]) > maxy: return None
        best = None
        for (a, b, n) in self.segments:
            r = seg_seg_intersect(p_prev, p_now, a, b)
            if r is None: continue
            hp, t = r
            if best is None or t < best[2]:
                best = (hp, n, t)
        return best

# ---------- rigid body ----------
# NOTE: in this desktop port "down" = +y (pygame). Gravity pulls toward
# larger y; the ground sits at world_h - GROUND_MARGIN; "lowest point" is the
# vertex with the LARGEST y. All three were flipped from the iOS version.
class RigidBody:
    def __init__(self, shape, world_h):
        cx, cy = shape.centroid
        self.local = [(p[0]-cx, p[1]-cy) for p in shape.pts]
        self.cx = cx
        self.cy = cy
        self.theta = 0.0
        self.omega = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.area = shape.area
        self.width = shape.width
        self.height = shape.height
        self.char_length = shape.char_length
        self.mass = max(BODY_DENSITY * shape.area, 0.05)
        r2 = 0.0
        for (lx, ly) in self.local:
            d = lx*lx + ly*ly
            if d > r2: r2 = d
        self.inertia = max(0.5 * self.mass * r2, 1.0)
        self.phase = 'falling'
        self.ground_y = world_h - GROUND_MARGIN   # floor in screen space
        self.settled = False

    def world_pts(self):
        c = math.cos(self.theta); s = math.sin(self.theta)
        out = []
        for (lx, ly) in self.local:
            wx = self.cx + lx*c - ly*s
            wy = self.cy + lx*s + ly*c
            out.append((wx, wy))
        return out

    def to_shape(self):
        return Shape(self.world_pts())

    def lowest_point(self):
        """Lowest = largest y in screen space."""
        pts = self.world_pts()
        ymax = -1e18; xat = self.cx
        for (x, y) in pts:
            if y > ymax:
                ymax = y; xat = x
        return ymax, xat

    def footprint(self):
        pts = self.world_pts()
        ymax = max(p[1] for p in pts)
        contact = [p for p in pts if p[1] > ymax - 8.0]
        if not contact:
            return self.width * 0.5, self.cx
        xs = [p[0] for p in contact]
        cxs = sum(xs)/len(xs)
        half = max((max(xs)-min(xs))*0.5, self.width*0.25, 4.0)
        return half, cxs

    def step_falling(self, dt):
        self.vy += GRAVITY * dt          # gravity pulls down (+y)
        self.cx += self.vx * dt
        self.cy += self.vy * dt
        self.theta += self.omega * dt
        self.omega *= ANG_DAMP

        ymax, xat = self.lowest_point()
        if ymax >= self.ground_y:
            overshoot = ymax - self.ground_y
            self.cy -= overshoot
            if self.vy > 0:
                self.vy = -self.vy * REST_DAMP
            self.vx *= GROUND_FRICTION
            self.omega *= 0.5
            # settle test: once the post-bounce speed is small enough we grab it.
            # compare against the *rebound* speed (post-damp) so a body that's
            # done bouncing doesn't micro-jitter forever just above the floor.
            if abs(self.vy) < SETTLE_EPS * 1.5 and abs(self.vx) < SETTLE_EPS:
                self.vy = 0.0; self.vx = 0.0; self.omega = 0.0
                self.phase = 'grounded'
                self.settled = True

    def step_grounded(self, dt, F_drag, wind_dir):
        half_w, contact_cx = self.footprint()
        h_above = max(self.cy - self.ground_y + self.height*0.5, self.height*0.5)
        h_eff = h_above * 0.5

        M_wind = F_drag * h_eff
        M_grav = self.mass * GRAVITY * half_w

        if M_grav <= 1e-6:
            ratio = 999.0
        else:
            ratio = M_wind / M_grav

        if ratio > RIP_MARGIN:
            self.phase = 'torn'
            speed = math.sqrt(max(F_drag / self.mass, 0.0)) * 6.0 + 60.0
            self.vx = wind_dir[0] * speed
            self.vy = wind_dir[1] * speed - 120.0   # initial upward lift (-y)
            self.omega = TORN_SPIN * (1.0 if wind_dir[0] >= 0 else -1.0)
            return True
        return False

    def step_torn(self, dt, sample_vel, world_w, world_h):
        u, v = sample_vel(self.cx, self.cy, world_w, world_h)
        if u != u: u = 0.0
        if v != v: v = 0.0
        relax = min(1.0, dt * 3.0)
        self.vx += (u - self.vx) * relax
        self.vy += (v - self.vy) * relax
        self.vy -= 40.0 * dt          # mild lift (upward = -y)
        self.cx += self.vx * dt
        self.cy += self.vy * dt
        self.theta += self.omega * dt
        self.omega *= 0.995

    def step(self, dt, F_drag, wind_dir, sample_vel, world_w, world_h):
        ripped = False
        if self.phase == 'falling':
            self.step_falling(dt)
        elif self.phase == 'grounded':
            ripped = self.step_grounded(dt, F_drag, wind_dir)
        elif self.phase == 'torn':
            self.step_torn(dt, sample_vel, world_w, world_h)
        return ripped

# ---------- particles ----------
class Particle:
    __slots__ = ('x','y','vx','vy','life','trail','alive','kind')
    def __init__(self, x, y, vx, vy, kind='laminar'):
        self.x = x; self.y = y
        self.vx = vx; self.vy = vy
        self.life = PARTICLE_LIFETIME
        self.trail = []
        self.alive = True
        self.kind = kind

class BadMarker:
    __slots__ = ('x','y','life')
    def __init__(self, x, y):
        self.x = x; self.y = y; self.life = BAD_MARKER_LIFE

# ---------- fluid grid (Stam-style semi-Lagrangian) ----------
class FluidGrid:
    def __init__(self, nx, ny):
        self.nx = nx; self.ny = ny
        n = nx*ny
        self.u  = [0.0]*n
        self.v  = [0.0]*n
        self.u0 = [0.0]*n
        self.v0 = [0.0]*n
        self.p  = [0.0]*n
        self.div = [0.0]*n
        self.solid = [0]*n
        self.pressure_field = [0.0]*n

    def idx(self, i, j): return j*self.nx + i

    def clear_solid(self):
        for k in range(len(self.solid)): self.solid[k] = 0

    def rasterize_shape(self, shape, world_w, world_h):
        self.clear_solid()
        if shape is None or not shape.valid: return
        nx, ny = self.nx, self.ny
        cw = world_w / nx
        ch = world_h / ny
        minx, miny, maxx, maxy = shape.bbox
        i0 = max(0, int(minx/cw) - 1)
        i1 = min(nx-1, int(maxx/cw) + 1)
        j0 = max(0, int(miny/ch) - 1)
        j1 = min(ny-1, int(maxy/ch) + 1)
        for j in range(j0, j1+1):
            for i in range(i0, i1+1):
                cx = (i+0.5)*cw
                cy = (j+0.5)*ch
                if point_in_polygon((cx, cy), shape.pts):
                    self.solid[self.idx(i,j)] = 1

    def set_inflow(self, ux_grid, uy_grid):
        nx, ny = self.nx, self.ny
        if ux_grid >= 0:
            for j in range(ny):
                self.u[j*nx + 0] = ux_grid
                self.v[j*nx + 0] = uy_grid
                self.u[j*nx + (nx-1)] = self.u[j*nx + (nx-2)]
                self.v[j*nx + (nx-1)] = self.v[j*nx + (nx-2)]
        else:
            for j in range(ny):
                self.u[j*nx + (nx-1)] = ux_grid
                self.v[j*nx + (nx-1)] = uy_grid
                self.u[j*nx + 0] = self.u[j*nx + 1]
                self.v[j*nx + 0] = self.v[j*nx + 1]

        if uy_grid >= 0:
            for i in range(nx):
                self.u[0*nx + i] = ux_grid
                self.v[0*nx + i] = uy_grid
                self.u[(ny-1)*nx + i] = self.u[(ny-2)*nx + i]
                self.v[(ny-1)*nx + i] = self.v[(ny-2)*nx + i]
        else:
            for i in range(nx):
                self.u[(ny-1)*nx + i] = ux_grid
                self.v[(ny-1)*nx + i] = uy_grid
                self.u[0*nx + i] = self.u[1*nx + i]
                self.v[0*nx + i] = self.v[1*nx + i]

    def _enforce_solid(self):
        for k in range(len(self.solid)):
            if self.solid[k]:
                self.u[k] = 0.0
                self.v[k] = 0.0

    def _sample(self, field, x, y):
        nx, ny = self.nx, self.ny
        if x != x: x = nx*0.5
        if y != y: y = ny*0.5
        if x < 0.5: x = 0.5
        if y < 0.5: y = 0.5
        if x > nx-1.5: x = nx-1.5
        if y > ny-1.5: y = ny-1.5
        i0 = int(x); j0 = int(y)
        i1 = i0+1;   j1 = j0+1
        sx = x - i0; sy = y - j0
        a = field[j0*nx+i0]; b = field[j0*nx+i1]
        c = field[j1*nx+i0]; d = field[j1*nx+i1]
        return (a*(1-sx)+b*sx)*(1-sy) + (c*(1-sx)+d*sx)*sy

    def advect(self, dt):
        nx, ny = self.nx, self.ny
        u_old, v_old = self.u, self.v
        u_new = self.u0
        v_new = self.v0
        solid = self.solid
        sample = self._sample
        for j in range(ny):
            base = j*nx
            for i in range(nx):
                k = base + i
                if solid[k]:
                    u_new[k] = 0.0; v_new[k] = 0.0
                    continue
                uk = u_old[k]; vk = v_old[k]
                x = i - dt * uk
                y = j - dt * vk
                u_new[k] = sample(u_old, x, y)
                v_new[k] = sample(v_old, x, y)
        self.u, self.u0 = u_new, u_old
        self.v, self.v0 = v_new, v_old

    def diffuse(self, visc, dt):
        if visc <= 1e-6: return
        nx, ny = self.nx, self.ny
        a = dt * visc
        denom = 1.0 + 4.0*a
        u, v = self.u, self.v
        u0, v0 = self.u0, self.v0
        solid = self.solid
        for _ in range(DIFFUSE_ITERS):
            for j in range(1, ny-1):
                base = j*nx
                for i in range(1, nx-1):
                    k = base + i
                    if solid[k]: continue
                    u[k] = (u0[k] + a*(u[k-1]+u[k+1]+u[k-nx]+u[k+nx])) / denom
                    v[k] = (v0[k] + a*(v[k-1]+v[k+1]+v[k-nx]+v[k+nx])) / denom

    def project(self):
        nx, ny = self.nx, self.ny
        u, v = self.u, self.v
        p = self.p
        div = self.div
        solid = self.solid

        for j in range(1, ny-1):
            base = j*nx
            for i in range(1, nx-1):
                k = base + i
                if solid[k]:
                    div[k] = 0.0
                    continue
                u_l = 0.0 if solid[k-1]  else u[k-1]
                u_r = 0.0 if solid[k+1]  else u[k+1]
                v_d = 0.0 if solid[k-nx] else v[k-nx]
                v_u = 0.0 if solid[k+nx] else v[k+nx]
                div[k] = -0.5 * ((u_r - u_l) + (v_u - v_d))

        for _ in range(NS_ITERS):
            for j in range(1, ny-1):
                base = j*nx
                for i in range(1, nx-1):
                    k = base + i
                    if solid[k]: continue
                    pl = p[k-1]  if not solid[k-1]  else p[k]
                    pr = p[k+1]  if not solid[k+1]  else p[k]
                    pd = p[k-nx] if not solid[k-nx] else p[k]
                    pu = p[k+nx] if not solid[k+nx] else p[k]
                    p[k] = (div[k] + pl + pr + pd + pu) * 0.25

        for j in range(1, ny-1):
            base = j*nx
            for i in range(1, nx-1):
                k = base + i
                if solid[k]: continue
                pl = p[k-1]  if not solid[k-1]  else p[k]
                pr = p[k+1]  if not solid[k+1]  else p[k]
                pd = p[k-nx] if not solid[k-nx] else p[k]
                pu = p[k+nx] if not solid[k+nx] else p[k]
                u[k] -= 0.5 * (pr - pl)
                v[k] -= 0.5 * (pu - pd)

        for k in range(nx*ny):
            self.pressure_field[k] = p[k]

    def step(self, dt, visc, ux_grid, uy_grid):
        speed = max(abs(ux_grid), abs(uy_grid), 1e-3)
        max_dt = 1.0 / speed
        substeps = max(1, min(8, int(math.ceil(dt / max_dt))))
        sub_dt = dt / substeps

        for _ in range(substeps):
            self.set_inflow(ux_grid, uy_grid)
            self._enforce_solid()
            self.advect(sub_dt)
            for k in range(len(self.u)):
                self.u0[k] = self.u[k]; self.v0[k] = self.v[k]
            self.diffuse(visc, sub_dt)
            self._enforce_solid()
            self.project()
            self._enforce_solid()

    def sample_velocity(self, x, y, world_w, world_h):
        gx = x / world_w * self.nx
        gy = y / world_h * self.ny
        u = self._sample(self.u, gx, gy)
        v = self._sample(self.v, gx, gy)
        return u * world_w / self.nx, v * world_h / self.ny

    def compute_forces(self, shape, world_w, world_h, wind_dir, U_world, rho=1.0):
        if shape is None or not shape.valid: return 0.0, 0.0
        nx, ny = self.nx, self.ny
        Fx = 0.0; Fy = 0.0
        solid = self.solid
        p = self.p
        for j in range(1, ny-1):
            base = j*nx
            for i in range(1, nx-1):
                k = base + i
                if not solid[k]: continue
                for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
                    ni, nj = i+di, j+dj
                    nk = nj*nx+ni
                    if 0 <= ni < nx and 0 <= nj < ny and not solid[nk]:
                        p_fluid = p[nk]
                        Fx -= p_fluid * di
                        Fy -= p_fluid * dj
        L_grid = shape.char_length * nx / max(world_w, 1.0)
        U_grid = U_world * nx / max(world_w, 1.0)
        dyn = 0.5 * rho * U_grid * U_grid * max(L_grid, 1.0)
        if dyn < 1e-6: return 0.0, 0.0
        CALIB = 180.0
        wdx, wdy = wind_dir
        Cd = CALIB * (Fx*wdx + Fy*wdy) / dyn
        Cl = CALIB * (-Fx*wdy + Fy*wdx) / dyn
        if Cd != Cd: Cd = 0.0
        if Cl != Cl: Cl = 0.0
        return Cd, Cl

# ======================================================================
#  SIMULATION  (engine state, no rendering)
# ======================================================================
class WindTunnel:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.particles = []
        self.bad_markers = []
        self.shape = None
        self.draw_points = []
        self.drawing = False

        self.wind_speed = 220.0
        self.density = 1.0
        self.wind_angle_deg = 0.0
        self.particle_size = 2.5
        self.show_streamlines = False
        self.show_heatmap = False
        self.mode = 'draw'

        self.heat = [[0.0]*HEAT_H for _ in range(HEAT_W)]

        self.cd_estimate = 0.0
        self.cl_estimate = 0.0
        self.reynolds = 0.0
        self.regime = 'LAMINAR'

        self._spawn_acc = 0.0
        self.fluid = FluidGrid(FX, FY)
        self._solid_dirty = True
        self.kinematic_viscosity = 8.0

        self._cd_avg = 0.0
        self._cl_avg = 0.0

        self.physics_on = False
        self.body = None
        self.body_status = 'OFF'

        self.weight_value = 0.0
        self.weight_unit = 'кг'
        self.weight_grams = 0.0

    @property
    def size(self):
        return (self.w, self.h)

    # canvas region is everything right of the sidebar
    def in_canvas(self, x, y):
        return x >= SIDEBAR_W

    def _wind_dir(self):
        a = math.radians(self.wind_angle_deg)
        return (math.cos(a), math.sin(a))

    def begin_draw(self, x, y):
        if self.mode != 'draw': return
        if self.physics_on: return
        if not self.in_canvas(x, y): return
        self.drawing = True
        self.draw_points = [(x, y)]

    def move_draw(self, x, y):
        if not self.drawing: return
        last = self.draw_points[-1]
        if (x-last[0])**2 + (y-last[1])**2 > 25:
            self.draw_points.append((x, y))

    def end_draw(self):
        if not self.drawing: return
        self.drawing = False
        if len(self.draw_points) >= 3:
            self.shape = Shape(self.draw_points[:])
            self.bad_markers = []
            self._solid_dirty = True
        self.draw_points = []

    def toggle_physics(self):
        if not self.physics_on:
            if self.shape is None or not self.shape.valid:
                return False
            self.body = RigidBody(self.shape, self.h)
            if self.weight_grams > 0:
                self.body.mass = max(self.weight_grams * 8e-5, 0.05)
                r2 = 0.0
                for (lx, ly) in self.body.local:
                    d = lx*lx + ly*ly
                    if d > r2: r2 = d
                self.body.inertia = max(0.5 * self.body.mass * r2, 1.0)
            self.physics_on = True
            self.body_status = 'FALLING'
            return True
        else:
            self.physics_on = False
            if self.body is not None:
                self.shape = self.body.to_shape()
                self._solid_dirty = True
            self.body = None
            self.body_status = 'OFF'
            return True

    def set_weight(self, val, unit):
        mult = {'г': 1.0, 'кг': 1000.0, 'т': 1_000_000.0}[unit]
        self.weight_value = val
        self.weight_unit = unit
        self.weight_grams = val * mult
        if self.physics_on and self.body is not None and self.weight_grams > 0:
            self.body.mass = max(self.weight_grams * 2e-4, 0.05)
            r2 = 0.0
            for (lx, ly) in self.body.local:
                d = lx*lx + ly*ly
                if d > r2: r2 = d
            self.body.inertia = max(0.5 * self.body.mass * r2, 1.0)

    def _spawn_tracers(self, dt):
        rate = SPAWN_BASE * self.density
        self._spawn_acc += rate * dt
        n = int(self._spawn_acc)
        self._spawn_acc -= n
        wd = self._wind_dir()
        w, h = self.size
        speed = self.wind_speed
        for _ in range(n):
            if len(self.particles) >= MAX_PARTICLES: break
            cx, cy = w*0.5, h*0.5
            perp = (-wd[1], wd[0])
            back = max(w, h) * 0.55
            offset = random.uniform(-h*0.6, h*0.6)
            sx = cx - wd[0]*back + perp[0]*offset
            sy = cy - wd[1]*back + perp[1]*offset
            vx = wd[0]*speed
            vy = wd[1]*speed
            self.particles.append(Particle(sx, sy, vx, vy, 'laminar'))

    def _drag_force_world(self):
        if self.shape is None or not self.shape.valid:
            return 0.0
        rho = self.density
        Cd = abs(self.cd_estimate)
        if Cd < 0.05: Cd = 0.05
        A = max(self.shape.height, self.shape.char_length, 1.0)
        v = self.wind_speed
        return 0.5 * rho * Cd * A * v * v * 3e-3

    def _step_fluid(self, dt):
        w, h = self.size
        wd = self._wind_dir()
        U_grid_x = wd[0] * self.wind_speed * FX / max(w, 1.0)
        U_grid_y = wd[1] * self.wind_speed * FY / max(h, 1.0)

        if self._solid_dirty:
            self.fluid.rasterize_shape(self.shape, w, h)
            self._solid_dirty = False

        nu_grid = self.kinematic_viscosity * 0.0008
        self.fluid.step(dt, nu_grid, U_grid_x, U_grid_y)

        L = self.shape.char_length if (self.shape and self.shape.valid) else 100.0
        nu_world = self.kinematic_viscosity
        self.reynolds = self.wind_speed * L / max(nu_world, 1e-3)
        if   self.reynolds < 100:    self.regime = 'LAMINAR'
        elif self.reynolds < 2000:   self.regime = 'TRANSITIONAL'
        else:                        self.regime = 'TURBULENT'

        cd, cl = self.fluid.compute_forces(self.shape, w, h, wd, self.wind_speed)
        alpha = min(1.0, dt * 2.0)
        self._cd_avg += (cd - self._cd_avg) * alpha
        self._cl_avg += (cl - self._cl_avg) * alpha
        self.cd_estimate = self._cd_avg
        self.cl_estimate = self._cl_avg

    def _step_body(self, dt):
        if not self.physics_on or self.body is None:
            return
        w, h = self.size
        wd = self._wind_dir()
        F_drag = self._drag_force_world()
        ripped = self.body.step(dt, F_drag, wd, self.fluid.sample_velocity, w, h)

        ph = self.body.phase
        if ph == 'falling':
            self.body_status = 'FALLING'
        elif ph == 'grounded':
            self.body_status = 'HOLDING'
        elif ph == 'torn':
            self.body_status = 'TORN OFF'

        if ripped:
            self._burst_particles(self.body.cx, self.body.cy, wd)

        self.shape = self.body.to_shape()
        if ph in ('falling', 'grounded'):
            self._solid_dirty = True
        elif ph == 'torn':
            self.fluid.clear_solid()

    def _burst_particles(self, x, y, wd):
        for _ in range(60):
            if len(self.particles) >= MAX_PARTICLES: break
            ang = random.uniform(0, 2*math.pi)
            sp = random.uniform(40, self.wind_speed*0.8)
            vx = wd[0]*self.wind_speed*0.5 + math.cos(ang)*sp
            vy = wd[1]*self.wind_speed*0.5 + math.sin(ang)*sp
            self.particles.append(Particle(x, y, vx, vy, 'debris'))

    def _step_particles(self, dt):
        w, h = self.size
        wd = self._wind_dir()
        collide_shape = self.shape
        if self.physics_on and self.body is not None and self.body.phase == 'torn':
            collide_shape = None

        new_list = []
        for p in self.particles:
            if not p.alive: continue
            p.life -= dt
            if p.life <= 0: continue

            world_vx, world_vy = self.fluid.sample_velocity(p.x, p.y, w, h)
            if world_vx != world_vx: world_vx = wd[0]*self.wind_speed
            if world_vy != world_vy: world_vy = wd[1]*self.wind_speed

            relax = min(1.0, dt * 6.0)
            p.vx += (world_vx - p.vx) * relax
            p.vy += (world_vy - p.vy) * relax

            px_prev, py_prev = p.x, p.y
            p.x += p.vx * dt
            p.y += p.vy * dt

            if collide_shape and collide_shape.valid:
                hit = collide_shape.collide((px_prev, py_prev), (p.x, p.y))
                if hit is not None:
                    hp, n, t = hit
                    v_in = v_norm((p.vx, p.vy))
                    cosang = abs(v_dot(v_in, n))
                    ang_from_perp_deg = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
                    if ang_from_perp_deg < FLAT_THRESHOLD_DEG:
                        self.bad_markers.append(BadMarker(hp[0], hp[1]))
                        continue
                    else:
                        v = (p.vx, p.vy)
                        dn = v_dot(v, n)
                        rvx = v[0] - 2*dn*n[0]
                        rvy = v[1] - 2*dn*n[1]
                        p.vx = rvx * 0.85
                        p.vy = rvy * 0.85
                        p.x = hp[0] + n[0]*1.5
                        p.y = hp[1] + n[1]*1.5

            if collide_shape and collide_shape.valid and point_in_polygon((p.x,p.y), collide_shape.pts):
                continue

            margin = 80
            if p.x < -margin or p.x > w+margin or p.y < -margin or p.y > h+margin:
                continue

            p.trail.append((px_prev, py_prev))
            if len(p.trail) > TRAIL_LEN:
                p.trail.pop(0)

            new_list.append(p)
        self.particles = new_list

    def _step_heatmap(self, dt):
        if not self.show_heatmap: return
        mx = 1e-3
        pf = self.fluid.pressure_field
        for v in pf:
            av = abs(v)
            if av > mx: mx = av
        inv = 1.0 / mx
        for i in range(FX):
            for j in range(FY):
                self.heat[i][j] = pf[j*FX + i] * inv

    def _step_markers(self, dt):
        new = []
        for m in self.bad_markers:
            m.life -= dt
            if m.life > 0: new.append(m)
        self.bad_markers = new

    def update(self, dt):
        if dt > 0.05: dt = 0.05
        self._spawn_tracers(dt)
        self._step_fluid(dt)
        self._step_body(dt)
        self._step_particles(dt)
        self._step_heatmap(dt)
        self._step_markers(dt)

    def clear_bad(self):
        self.bad_markers = []

    def clear_shape(self):
        self.shape = None
        self.bad_markers = []
        self._solid_dirty = True
        self.physics_on = False
        self.body = None
        self.body_status = 'OFF'

    # ---- image obstacle (marching squares on alpha) ----
    def load_image_obstacle(self, pil_img):
        if not HAVE_PIL:
            return False
        try:
            img = pil_img.convert('RGBA')
            max_dim = 220
            w0, h0 = img.size
            scale = max_dim / max(w0, h0)
            if scale < 1.0:
                img = img.resize((int(w0 * scale), int(h0 * scale)))
            w0, h0 = img.size
            alpha = img.split()[3]
            mask = alpha.point(lambda v: 1 if v > 64 else 0)
            mp = mask.load()

            def m(x, y):
                if x < 0 or y < 0 or x >= w0 or y >= h0:
                    return 0
                return mp[x, y]

            segs = {}
            def E_top(x, y):    return (x * 2 + 1, y * 2)
            def E_bottom(x, y): return (x * 2 + 1, y * 2 + 2)
            def E_left(x, y):   return (x * 2,     y * 2 + 1)
            def E_right(x, y):  return (x * 2 + 2, y * 2 + 1)
            def add(a, b): segs[a] = b

            for y in range(-1, h0):
                for x in range(-1, w0):
                    tl = m(x, y); tr = m(x + 1, y)
                    br = m(x + 1, y + 1); bl = m(x, y + 1)
                    code = (tl << 3) | (tr << 2) | (br << 1) | bl
                    T, R, B, L = E_top(x, y), E_right(x, y), E_bottom(x, y), E_left(x, y)
                    if code in (1,):           add(B, L)
                    elif code in (2,):         add(R, B)
                    elif code in (3,):         add(R, L)
                    elif code in (4,):         add(T, R)
                    elif code == 5:            add(T, L); add(B, R)
                    elif code in (6,):         add(T, B)
                    elif code in (7,):         add(T, L)
                    elif code in (8,):         add(L, T)
                    elif code in (9,):         add(B, T)
                    elif code == 10:           add(L, B); add(R, T)
                    elif code in (11,):        add(R, T)
                    elif code in (12,):        add(L, R)
                    elif code in (13,):        add(B, R)
                    elif code in (14,):        add(L, B)

            if not segs:
                return False

            best_loop = None
            visited = set()
            for start in list(segs.keys()):
                if start in visited: continue
                loop = []; cur = start
                while cur in segs and cur not in visited:
                    visited.add(cur); loop.append(cur)
                    cur = segs[cur]
                    if cur == start: break
                if len(loop) >= 4 and (best_loop is None or len(loop) > len(best_loop)):
                    best_loop = loop

            if best_loop is None or len(best_loop) < 4:
                return False

            poly = [(p[0] * 0.5, p[1] * 0.5) for p in best_loop]

            def dp(points, eps):
                if len(points) < 3: return points
                dmax = 0.0; idx = 0
                ax, ay = points[0]; bx, by = points[-1]
                dx, dy = bx - ax, by - ay
                seglen = math.hypot(dx, dy) + 1e-9
                for i in range(1, len(points) - 1):
                    px, py = points[i]
                    d = abs((px - ax) * dy - (py - ay) * dx) / seglen
                    if d > dmax: dmax = d; idx = i
                if dmax > eps:
                    left = dp(points[:idx + 1], eps)
                    right = dp(points[idx:], eps)
                    return left[:-1] + right
                return [points[0], points[-1]]

            simplified = dp(poly, 1.2)
            if len(simplified) < 4:
                simplified = poly[::max(1, len(poly) // 60)]

            scw, sch = self.size
            xs = [p[0] for p in simplified]; ys = [p[1] for p in simplified]
            minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
            bw = maxx - minx; bh = maxy - miny
            target_w = scw * 0.35
            k = target_w / max(bw, 1)
            cx = scw * 0.55; cy = sch * 0.5
            # NOTE: no y-flip here (pygame is already y-down like image space)
            shifted = [
                ((p[0] - minx - bw * 0.5) * k + cx,
                 (p[1] - miny - bh * 0.5) * k + cy)
                for p in simplified
            ]
            self.shape = Shape(shifted)
            self.bad_markers = []
            self._solid_dirty = True
            return True
        except Exception as e:
            print('image obstacle err:', e)
            return False

# ======================================================================
#  PYGAME UI WIDGETS
# ======================================================================
class Widget:
    def __init__(self, rect):
        self.rect = pygame.Rect(rect)
    def handle(self, ev): return False
    def draw(self, surf, font): pass

class Slider(Widget):
    def __init__(self, rect, lo, hi, val, label, on_change):
        super().__init__(rect)
        self.lo = lo; self.hi = hi
        self.val = val
        self.label = label
        self.on_change = on_change
        self.dragging = False

    def _set_from_x(self, x):
        t = (x - self.rect.x) / max(self.rect.w, 1)
        t = max(0.0, min(1.0, t))
        self.val = self.lo + (self.hi - self.lo) * t
        self.on_change(self.val)

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(ev.pos):
            self.dragging = True
            self._set_from_x(ev.pos[0]); return True
        elif ev.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif ev.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(ev.pos[0]); return True
        return False

    def draw(self, surf, font):
        lab = font.render(f'{self.label}', True, (216, 230, 255))
        surf.blit(lab, (self.rect.x, self.rect.y - 16))
        pygame.draw.rect(surf, (40, 48, 64), (self.rect.x, self.rect.centery-2, self.rect.w, 4), border_radius=2)
        t = (self.val - self.lo) / (self.hi - self.lo) if self.hi > self.lo else 0
        kx = int(self.rect.x + t * self.rect.w)
        pygame.draw.circle(surf, (120, 200, 255), (kx, self.rect.centery), 8)

class Toggle(Widget):
    def __init__(self, rect, val, label, on_change):
        super().__init__(rect)
        self.val = val
        self.label = label
        self.on_change = on_change

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(ev.pos):
            self.val = not self.val
            self.on_change(self.val); return True
        return False

    def draw(self, surf, font):
        col = (40, 200, 120) if self.val else (70, 70, 80)
        pygame.draw.rect(surf, col, self.rect, border_radius=14)
        kx = self.rect.right - 14 if self.val else self.rect.x + 14
        pygame.draw.circle(surf, (240, 240, 245), (kx, self.rect.centery), 11)
        lab = font.render(self.label, True, (230, 230, 235))
        surf.blit(lab, (self.rect.right + 10, self.rect.centery - 8))

class Button(Widget):
    def __init__(self, rect, title, on_click, bg=(40, 60, 90)):
        super().__init__(rect)
        self.title = title
        self.on_click = on_click
        self.bg = bg

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(ev.pos):
            self.on_click(); return True
        return False

    def draw(self, surf, font):
        pygame.draw.rect(surf, self.bg, self.rect, border_radius=7)
        lab = font.render(self.title, True, (240, 240, 245))
        surf.blit(lab, (self.rect.centerx - lab.get_width()//2,
                        self.rect.centery - lab.get_height()//2))

class Segmented(Widget):
    def __init__(self, rect, segments, idx, on_change):
        super().__init__(rect)
        self.segments = segments
        self.idx = idx
        self.on_change = on_change

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(ev.pos):
            sw = self.rect.w / len(self.segments)
            i = int((ev.pos[0] - self.rect.x) / sw)
            i = max(0, min(len(self.segments)-1, i))
            self.idx = i
            self.on_change(i); return True
        return False

    def draw(self, surf, font):
        sw = self.rect.w / len(self.segments)
        for i, seg in enumerate(self.segments):
            r = pygame.Rect(self.rect.x + i*sw, self.rect.y, sw, self.rect.h)
            bg = (60, 110, 160) if i == self.idx else (35, 40, 55)
            pygame.draw.rect(surf, bg, r, border_radius=5)
            lab = font.render(seg, True, (235, 235, 240))
            surf.blit(lab, (r.centerx - lab.get_width()//2, r.centery - lab.get_height()//2))

# ======================================================================
#  RENDERER + MAIN LOOP
# ======================================================================
class App:
    def __init__(self, w=1024, h=720):
        pygame.init()
        pygame.display.set_caption('Wind Tunnel')
        self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.w, self.h = w, h
        self.sim = WindTunnel(w, h)
        self.font = pygame.font.SysFont('menlo,consolas,monospace', 13)
        self.font_b = pygame.font.SysFont('menlo,consolas,monospace', 14, bold=True)
        self.widgets = []
        self.btn_phys = None
        self.weight_text = ''
        self.weight_unit_idx = 1
        self._build_widgets()
        self.running = True

    def _build_widgets(self):
        self.widgets = []
        y = 12
        seg = Segmented((10, y, 220, 28), ['DRAW', 'IMAGE'], 0, self._on_mode)
        self.widgets.append(seg); y += 38

        # title drawn manually in render
        y += 24

        s = self.sim
        self.widgets.append(Slider((12, y, 216, 22), 50, 1500, s.wind_speed, 'Wind speed', lambda v: setattr(s, 'wind_speed', v))); y += 40
        self.widgets.append(Slider((12, y, 216, 22), 0.1, 2.0, s.density, 'Density', lambda v: setattr(s, 'density', v))); y += 40
        self.widgets.append(Slider((12, y, 216, 22), 0, 360, s.wind_angle_deg, 'Wind angle', lambda v: setattr(s, 'wind_angle_deg', v))); y += 40
        self.widgets.append(Slider((12, y, 216, 22), 1.0, 6.0, s.particle_size, 'Particle size', lambda v: setattr(s, 'particle_size', v))); y += 40
        self.widgets.append(Slider((12, y, 216, 22), 1.0, 60.0, s.kinematic_viscosity, 'Viscosity', lambda v: setattr(s, 'kinematic_viscosity', v))); y += 40

        self.widgets.append(Toggle((12, y, 44, 24), False, 'Streamlines', lambda v: setattr(s, 'show_streamlines', v))); y += 32
        self.widgets.append(Toggle((12, y, 44, 24), False, 'Pressure heatmap', lambda v: setattr(s, 'show_heatmap', v))); y += 40

        # weight row
        self.weight_label_y = y; y += 22
        self.weight_field_rect = pygame.Rect(12, y, 120, 28)
        self.seg_unit = Segmented((138, y, 92, 28), ['г', 'кг', 'т'], 1, self._on_unit)
        self.widgets.append(self.seg_unit); y += 38

        self.btn_phys = Button((12, y, 216, 34), 'ВКЛЮЧИТЬ ФИЗИКУ', self._toggle_phys, bg=(26, 76, 46)); 
        self.widgets.append(self.btn_phys); y += 42

        self.widgets.append(Button((12, y, 216, 30), 'Clear bad markers', s.clear_bad, bg=(64, 20, 26))); y += 36
        self.widgets.append(Button((12, y, 216, 30), 'Save screenshot', self._screenshot, bg=(26, 52, 76))); y += 36
        self.widgets.append(Button((12, y, 216, 30), 'Clear shape', self._clear_shape, bg=(38, 38, 46))); y += 36
        self.btn_pick = Button((12, y, 216, 30), 'Pick PNG (image mode)', self._pick_image, bg=(30, 64, 46))
        self.widgets.append(self.btn_pick); y += 40

        self.analytics_y = y

    def _on_mode(self, i):
        self.sim.mode = 'draw' if i == 0 else 'image'
        self.sim.clear_shape()
        self._sync_phys()

    def _on_unit(self, i):
        units = ['г', 'кг', 'т']
        self.weight_unit_idx = i
        self._recalc_weight()

    def _recalc_weight(self):
        try:
            val = float(self.weight_text.replace(',', '.')) if self.weight_text.strip() else 0.0
        except ValueError:
            val = 0.0
        unit = ['г', 'кг', 'т'][self.weight_unit_idx]
        self.sim.set_weight(val, unit)

    def _toggle_phys(self):
        ok = self.sim.toggle_physics()
        if not ok:
            self.btn_phys.title = 'СНАЧАЛА НАРИСУЙ ФОРМУ'
            return
        self._sync_phys()

    def _sync_phys(self):
        if self.sim.physics_on:
            self.btn_phys.title = 'ВЫКЛЮЧИТЬ ФИЗИКУ'
            self.btn_phys.bg = (90, 30, 30)
        else:
            self.btn_phys.title = 'ВКЛЮЧИТЬ ФИЗИКУ'
            self.btn_phys.bg = (26, 76, 46)

    def _clear_shape(self):
        self.sim.clear_shape()
        self._sync_phys()

    def _screenshot(self):
        i = 0
        while os.path.exists(f'wind_tunnel_{i}.png'):
            i += 1
        pygame.image.save(self.screen, f'wind_tunnel_{i}.png')
        print(f'saved wind_tunnel_{i}.png')

    def _pick_image(self):
        if not HAVE_PIL:
            print('PIL not installed - image mode unavailable')
            return
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            path = filedialog.askopenfilename(
                filetypes=[('Images', '*.png *.jpg *.jpeg *.bmp *.gif')])
            root.destroy()
            if not path:
                return
            pil = Image.open(path)
            ok = self.sim.load_image_obstacle(pil)
            if not ok:
                print('failed to build silhouette (need image with transparency or clear shape)')
        except Exception as e:
            print('pick err:', e)

    # ---- input ----
    def handle_event(self, ev):
        if ev.type == pygame.QUIT:
            self.running = False
            return
        if ev.type == pygame.VIDEORESIZE:
            self.w, self.h = ev.w, ev.h
            self.sim.w, self.sim.h = ev.w, ev.h
            self.screen = pygame.display.set_mode((ev.w, ev.h), pygame.RESIZABLE)
            return
        if ev.type == pygame.KEYDOWN:
            # weight text field editing (always active, simple)
            if ev.key == pygame.K_BACKSPACE:
                self.weight_text = self.weight_text[:-1]; self._recalc_weight()
            elif ev.unicode and (ev.unicode.isdigit() or ev.unicode in '.,'):
                self.weight_text += ev.unicode; self._recalc_weight()

        # widgets first (sidebar)
        if ev.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
            mx = ev.pos[0] if hasattr(ev, 'pos') else 0
            if mx < SIDEBAR_W:
                for wdg in self.widgets:
                    if wdg.handle(ev):
                        return
                return  # consumed by sidebar area

        # canvas drawing
        if ev.type == pygame.MOUSEBUTTONDOWN:
            self.sim.begin_draw(*ev.pos)
        elif ev.type == pygame.MOUSEMOTION:
            if ev.buttons[0]:
                self.sim.move_draw(*ev.pos)
        elif ev.type == pygame.MOUSEBUTTONUP:
            self.sim.end_draw()

    # ---- render ----
    def render(self):
        s = self.sim
        scr = self.screen
        scr.fill((10, 13, 20))

        # ground line when physics on
        if s.physics_on:
            gy = s.h - GROUND_MARGIN
            pygame.draw.line(scr, (102, 89, 64), (SIDEBAR_W, gy), (s.w, gy), 2)
            pygame.draw.rect(scr, (26, 23, 18), (SIDEBAR_W, gy, s.w-SIDEBAR_W, GROUND_MARGIN))

        # heatmap
        if s.show_heatmap:
            cw = s.w / HEAT_W; ch = s.h / HEAT_H
            for i in range(HEAT_W):
                for j in range(HEAT_H):
                    pv = s.heat[i][j]
                    if abs(pv) < 0.05: continue
                    if pv > 0:
                        col = (255, 76, 51); a = min(0.45, pv*0.5)
                    else:
                        col = (51, 128, 255); a = min(0.45, -pv*0.5)
                    surf = pygame.Surface((int(cw)+1, int(ch)+1), pygame.SRCALPHA)
                    surf.fill((col[0], col[1], col[2], int(a*255)))
                    scr.blit(surf, (int(i*cw), int(j*ch)))

        # shape
        if s.shape and s.shape.valid:
            if s.physics_on and s.body is not None:
                if s.body.phase == 'torn':      fill = (90, 46, 31)
                elif s.body.phase == 'grounded':fill = (51, 51, 38)
                else:                           fill = (38, 46, 56)
            else:
                fill = (38, 46, 56)
            pts = [(int(p[0]), int(p[1])) for p in s.shape.pts]
            if len(pts) >= 3:
                pygame.draw.polygon(scr, fill, pts)
                pygame.draw.polygon(scr, (128, 217, 255), pts, 2)

        # drawing in progress
        if s.drawing and len(s.draw_points) > 1:
            pts = [(int(p[0]), int(p[1])) for p in s.draw_points]
            pygame.draw.lines(scr, (204, 230, 255), False, pts, 2)

        # streamlines
        if s.show_streamlines:
            for k, p in enumerate(s.particles):
                if k % 6 != 0: continue
                if len(p.trail) < 2: continue
                pts = [(int(a[0]), int(a[1])) for a in p.trail]
                pygame.draw.lines(scr, (102, 204, 255), False, pts, 1)

        # particles
        sz = s.particle_size
        for p in s.particles:
            speed = math.hypot(p.vx, p.vy)
            t = min(1.0, speed / max(s.wind_speed, 1.0))
            if p.kind == 'debris':  col = (255, 204, 115)
            elif t < 0.4:           col = (255, 153, 230)
            else:                   col = (217, 255, 255)
            pygame.draw.circle(scr, col, (int(p.x), int(p.y)), max(1, int(sz)))

        # bad markers
        for m in s.bad_markers:
            a = max(0.0, m.life / BAD_MARKER_LIFE)
            col = (255, int(51*a)+30, 64)
            r = 7
            pygame.draw.line(scr, col, (m.x-r, m.y-r), (m.x+r, m.y+r), 2)
            pygame.draw.line(scr, col, (m.x-r, m.y+r), (m.x+r, m.y-r), 2)

        # wind dir indicator
        wd = s._wind_dir()
        ox, oy = s.w-60, 60
        pygame.draw.line(scr, (179, 255, 255), (ox, oy), (ox+wd[0]*30, oy+wd[1]*30), 2)
        pygame.draw.circle(scr, (179, 255, 255), (int(ox+wd[0]*30), int(oy+wd[1]*30)), 4)

        # sidebar
        self._render_sidebar()
        pygame.display.flip()

    def _render_sidebar(self):
        s = self.sim
        scr = self.screen
        pygame.draw.rect(scr, (15, 18, 26), (0, 0, SIDEBAR_W, s.h))

        # title
        title = self.font_b.render('WIND TUNNEL', True, (153, 242, 255))
        scr.blit(title, (12, 46))

        for wdg in self.widgets:
            wdg.draw(scr, self.font)

        # weight label + field
        wl = self.font.render('Вес объекта', True, (216, 230, 255))
        scr.blit(wl, (12, self.weight_label_y))
        pygame.draw.rect(scr, (30, 34, 44), self.weight_field_rect, border_radius=4)
        pygame.draw.rect(scr, (70, 80, 100), self.weight_field_rect, 1, border_radius=4)
        txt = self.weight_text if self.weight_text else '0 = из площади'
        col = (235,235,240) if self.weight_text else (120,125,140)
        wf = self.font.render(txt, True, col)
        scr.blit(wf, (self.weight_field_rect.x+6, self.weight_field_rect.y+6))

        # analytics
        y = self.analytics_y
        def line(txt, col):
            nonlocal y
            scr.blit(self.font.render(txt, True, col), (12, y)); y += 22
        re = s.reynolds
        re_txt = f'Re = {re:.2e}' if re > 1e4 else f'Re = {re:.0f}'
        line(f'Cd = {s.cd_estimate:+.2f}', (230, 242, 255))
        line(f'Cl = {s.cl_estimate:+.2f}', (242, 230, 179))
        line(re_txt, (204, 242, 255))
        line(f'Regime: {s.regime}', (179, 255, 204))
        line(f'Stagnation pts: {len(s.bad_markers)}', (255, 153, 153))
        line(f'Body: {s.body_status}', (255, 217, 102))

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            for ev in pygame.event.get():
                self.handle_event(ev)
            self.sim.update(dt)
            self.render()
        pygame.quit()

def main():
    App().run()

if __name__ == '__main__':
    main()
