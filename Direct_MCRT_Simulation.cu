#include "SF_mcrt_CPC.cuh"
#include "vector.cuh"
#include "TMS.cuh"
#include "bezier.cuh"
#include "gaussian_filter.cuh"
// #include "sample_vMF.cuh"
#include "cyli_reflectors.cuh"
#include "sun_shape.cuh"

/*
    @brief 计算与二级反射镜的求交+反射
    @param origin 光线起点
    @param dir 光线方向
    @param reflector_origin 反射镜起点
    @param reflector_dir 反射镜方向
    @param t 光线传播距离
    @param local_state 随机数状态
    @param slope_error 斜率误差
    @return 是否命中
*/
__device__ bool intersect_secondary_reflector(
    const vec3f &origin,
    const vec3f &dir,
    vec3f &reflector_origin,
    vec3f &reflector_dir,
    float &t,
    curandState *local_state,
    float slope_error)
{
    if (dir.z <= 0.0f)
        return false;
    // 左/右 Bezier 的控制点（x,z）。y 恒为 0（曲线在 x-z 平面）。
    // 左侧
    const float lp0x = 0.0f, lp0z = p0z;    // 8.14
    const float lp1x = -0.075f, lp1z = p1z; // 8.14
    const float lp2x = -0.15f, lp2z = p2z;  // 7.90
    // 右侧
    const float rp0x = 0.0f, rp0z = p0z;
    const float rp1x = 0.075f, rp1z = p1z;
    const float rp2x = 0.15f, rp2z = p2z;

    // 先分别与左右曲线求交
    float t_b_l = 0, hx_l = 0, hz_l = 0, tray_l = 0;
    float t_b_r = 0, hx_r = 0, hz_r = 0, tray_r = 0;
    bool hitL = hit_bezier(origin, dir, lp0x, lp0z, lp1x, lp1z, lp2x, lp2z,
                           t_b_l, hx_l, hz_l, tray_l);
    bool hitR = hit_bezier(origin, dir, rp0x, rp0z, rp1x, rp1z, rp2x, rp2z,
                           t_b_r, hx_r, hz_r, tray_r);
    if (!hitL && !hitR)
        return false;

    // 选最近的交点（按光线参数 t_ray）
    bool useLeft = false;
    float t_ray = 0.0f, t_bezier = 0.0f, hitx = 0.0f, hitz = 0.0f;
    if (hitL && hitR)
    {
        if (tray_l <= tray_r)
        {
            useLeft = true;
            t_ray = tray_l;
            t_bezier = t_b_l;
            hitx = hx_l;
            hitz = hz_l;
        }
        else
        {
            useLeft = false;
            t_ray = tray_r;
            t_bezier = t_b_r;
            hitx = hx_r;
            hitz = hz_r;
        }
    }
    else if (hitL)
    {
        useLeft = true;
        t_ray = tray_l;
        t_bezier = t_b_l;
        hitx = hx_l;
        hitz = hz_l;
    }
    else
    {
        useLeft = false;
        t_ray = tray_r;
        t_bezier = t_b_r;
        hitx = hx_r;
        hitz = hz_r;
    }
    //  printf("origin=(%.4f,%.4f,%.4f),dir=(%.4f,%.4f,%.4f),hit_point:(%f,%f)\n",origin.x,origin.y,origin.z,dir.x,dir.y,dir.z,hitx,hitz);
    // 根据相同 t_ray 计算 y 坐标（反射器在 y 方向有限长）
    float y = origin.y + t_ray * dir.y;
    if (fabsf(y) > reflector_len * 0.5f)
        return false;

    // 命中点
    vec3f hitP = {hitx, y, hitz};

    // 曲线切向量 (dx, 0, dz)；法线设为 (dz, 0, -dx) 并做方向校正
    float dx = 0.0f, dz = 0.0f;
    if (useLeft)
    {
        dx = bezier_curve_derivative(lp0x, lp1x, lp2x, t_bezier);
        dz = bezier_curve_derivative(lp0z, lp1z, lp2z, t_bezier);
    }
    else
    {
        dx = bezier_curve_derivative(rp0x, rp1x, rp2x, t_bezier);
        dz = bezier_curve_derivative(rp0z, rp1z, rp2z, t_bezier);
    }
#ifndef SLOPE_ERROR
    vec3f N = normalize3({dz, 0.0f, -dx});
    // 若 dx < 0 则翻转法线，保证“朝外（向下）”
    if (dx < 0.0f)
        N = mul3(N, -1.0f);
#else
    vec3f N = normalize3({dz, 0.0f, -dx});
    if (dx < 0.0f)
        N = mul3(N, -1.0f);                     // 若 dx < 0 则翻转法线，保证“朝外（向下）”
    N = perturb_N(N, local_state, slope_error); // 对法线添加高斯分布斜率误差扰动
#endif
    // 镜面反射
    vec3f I = normalize3(dir);
    vec3f R = reflect3(I, N);

    reflector_origin = add3(hitP, mul3(R, EPSILON));
    reflector_dir = normalize3(R);
    t += t_ray;
    return true;
}

__device__ bool hit_parabola(
    const vec3f &origin,
    const vec3f &dir,
    float vertex_z, float vertex_x,
    float c,
    float x_min, float x_max,
    float &hit_x,
    float &hit_z,
    float &t_ray)
{

    float ox = origin.x;
    float oz = origin.z;
    float dx = dir.x;
    float dz = dir.z;
    float k = ox - vertex_x;

    // 3. 抛物线方程：在局部坐标系中，z_local = 0.5 * cx * x_local²

    // 整理为：A*t² + B*t + C = 0
    float A = c * dx * dx;
    float B = 2.0f * c * k * dx + dz;
    float C = c * k * k + oz - vertex_z;

    // 4. 解二次方程
    float discriminant = B * B - 4.0f * A * C;
    if (discriminant < 0.0f)
        return false;

    float sqrt_disc = sqrtf(discriminant);
    float t1, t2;

    if (fabsf(A) > 1e-12f)
    {
        t1 = (-B - sqrt_disc) / (2.0f * A);
        t2 = (-B + sqrt_disc) / (2.0f * A);
    }
    else
    {
        // A≈0，退化为线性方程
        if (fabsf(B) < 1e-12f)
            return false;
        t1 = t2 = -C / B;
    }

    // 5. 检查两个解，选择最近的有效的交点
    bool hit = false;
    float t_INF = 1e30f;

    auto check_solution = [&](float t)
    {
        if (t <= 0.0f || t >= t_INF)
            return; // 排除负解和更远的解

        // 计算命中点在局部坐标系的X/z坐标
        float x = ox + t * dx;
        float z = oz + t * dz;

        // 检查是否在抛物面X范围内
        if (x < x_min || x > x_max)
            return;

        hit = true;
        t_ray = t;
        hit_x = x;
        hit_z = z;
    };

    check_solution(t1);
    check_solution(t2);

    return hit;
}

/*
    @brief 计算与二级反射镜（抛物面）的求交+反射
    @param origin 光线起点
    @param dir 光线方向
    @param reflector_origin 反射镜起点（输出）
    @param reflector_dir 反射镜方向（输出）
    @param t 光线传播距离（累加）
    @param local_state 随机数状态
    @param slope_error 斜率误差
    @return 是否命中
*/
__device__ bool intersect_secondary_parabola(
    const vec3f &origin,
    const vec3f &dir,
    vec3f &reflector_origin,
    vec3f &reflector_dir,
    float &t,
    curandState *local_state,
    float slope_error,
    int refID,
    int count)
{
    if (dir.z <= 0.0f)
        return false;

    // 左侧抛物面参数
    const float vertex_z = 8.14f;
    const float f = 0.0215;
    const float c = 1.0f / (4.0f * f);

    // 左右两侧的瞄准点
    const float x_v_L = -0.00765f;
    const float x_v_R = 0.00765f;

    // X范围
    const float left_x_min = -0.15f, left_x_max = 0.0f;
    const float right_x_min = 0.0f, right_x_max = 0.15f;

    // 分别与左右抛物面求交
    float hit_x_l = 0.0f, hit_z_l = 0.0f, t_l = 0.0f;
    float hit_x_r = 0.0f, hit_z_r = 0.0f, t_r = 0.0f;

    bool hitL = hit_parabola(origin, dir, vertex_z, x_v_L, c,
                             left_x_min, left_x_max,
                             hit_x_l, hit_z_l, t_l);

    bool hitR = hit_parabola(origin, dir, vertex_z, x_v_R, c,
                             right_x_min, right_x_max,
                             hit_x_r, hit_z_r, t_r);

    if (!hitL && !hitR)
        return false;

    // 选择最近的交点
    bool useLeft = hitL && (!hitR || t_l < t_r);

    float hit_x = useLeft ? hit_x_l : hit_x_r;
    float hit_z = useLeft ? hit_z_l : hit_z_r;
    float t_ray = useLeft ? t_l : t_r;
    float x_v = useLeft ? x_v_L : x_v_R;

    float y = origin.y + t_ray * dir.y;
    if (fabsf(y) > reflector_len * 0.5f)
        return false;

    // 命中点
    vec3f hitP = {hit_x, y, hit_z};

    float dzdx = -2.0f * c * (hit_x - x_v);

    // 确保法线朝外（指向光线入射方向的反侧）
    vec3f N = normalize3({dzdx, 0.0f, -1.0f});
    vec3f to_surface = normalize3(sub3(hitP, origin));
    if (dot3(N, to_surface) < 0.0f)
    {
        N = mul3(N, -1.0f);
    }

    // 添加斜率误差
#ifndef SLOPE_ERROR
    // 不添加误差
#else
    N = perturb_N(N, local_state, slope_error);
#endif

    // 镜面反射
    vec3f I = normalize3(dir);
    vec3f R = reflect3(I, N);
    reflector_origin = add3(hitP, mul3(R, EPSILON));
    reflector_dir = normalize3(R);
    t += t_ray;

    return true;
}

// --------------------生成随机数--------------------
/*
    @brief 在设备上生成随机数的内核函数池
    @param state 随机数状态数组
    @param seed 随机数种子
    @param pool_size 随机数池大小
*/
__global__ void setup_kernel_pool(curandState *state, unsigned long seed, int pool_size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < pool_size)
    {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

/*
    @brief 从随机数池分配随机数给每个射线
    @param state 随机数状态数组
    @param ray_states 射线随机数状态数组
    @param nRays 射线数量
    @param pool_size 随机数池大小
*/
__global__ void assign_random_states(curandState *state, curandState *ray_states,
                                     int nRays, int pool_size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < nRays)
    {
        int pool_idx = idx % pool_size;
        ray_states[idx] = state[pool_idx];
        // 添加不同的偏移量
        for (int i = 0; i < (idx / pool_size); i++)
        {
            curand_uniform(&ray_states[idx]);
        }
    }
}

// --------------------玻璃管部分--------------------
/*
    @brief 基于斯涅尔定律计算折射方向
    @param I 入射方向
    @param N 法线方向
    @param n_i 入射介质折射率
    @param n_t 出射介质折射率
    @return 发生折射返回true，否则返回false
*/
__device__ __forceinline__ bool refract3(const vec3f &I, const vec3f &N, float n_i, float n_t, vec3f &refracted)
{
    bool entering = dot3(I, N) < 0.0f;
    vec3f normal = entering ? N : mul3(N, -1.0f);
    float eta = entering ? n_i / n_t : n_t / n_i;
    float cosi = -dot3(I, normal);
    float k = 1.0f - eta * eta * (1.0f - cosi * cosi);
    if (k < 0.0f) // 全反射
        return false;
    refracted = normalize3(add3(mul3(I, eta), mul3(N, eta * cosi - sqrtf(k))));
    return true;
}

/*
    @brief 计算菲涅尔反射率
    @param I 入射方向
    @param N 法线方向
    @param n_i 入射介质折射率
    @param n_t 出射介质折射率
    @return 反射概率
*/
__device__ __forceinline__ float fresnel(const vec3f &I, const vec3f &N, float n_i, float n_t)
{
    bool entering = dot3(I, N) < 0.0f;
    vec3f normal = entering ? N : mul3(N, -1.0f);
    float cosi = (-dot3(I, normal) < -1.0f) ? -1.0f : (-dot3(I, normal) > 1.0f) ? 1.0f
                                                                                : -dot3(I, normal);
    float eta = entering ? n_i / n_t : n_t / n_i;
    float sint = eta * sqrtf(1.0f - cosi * cosi);

    // 全反射情况
    if (sint >= 1.0f)
        return 1.0f;
    float cost = sqrtf(1.0f - sint * sint);
    float R_parallel = (eta * cosi - cost) / (eta * cosi + cost);
    float R_perpendicular = (cosi - eta * cost) / (cosi + eta * cost);

    // 平均反射率
    return (R_parallel * R_parallel + R_perpendicular * R_perpendicular) * 0.5f;
}

/*
    @brief 计算光线与圆的交点
    @param origin 射线起点(世界坐标系)
    @param direction 射线方向(世界坐标系)
    @param center 圆的中心(世界坐标系)
    @param radius 圆的半径
    @param hit_point 交点，若未命中则返回-1
    @param t 光线传播距离
    @return 交点坐标
*/
__device__ int ray_intersect_circle(
    const vec3f &origin,
    const vec3f &dir,
    const vec3f &center,
    const float radius,
    float &t,
    vec3f &hit_point)
{
    vec3f o = sub3(origin, center);
    float dx = dir.x, dz = dir.z;
    float ox = o.x, oz = o.z;

    float a = dx * dx + dz * dz;
    float b = 2.0f * (ox * dx + oz * dz);
    float c = ox * ox + oz * oz - radius * radius;

    if (a < EPSILON)
        return -1;

    float discriminant = b * b - 4.0f * a * c;
    if (discriminant < -EPSILON)
        return -1;
    if (discriminant < 0.0f)
        discriminant = 0.0f;
    float sqrt_disc = sqrtf(fmaxf(discriminant, 0.0f));
    float q = -0.5f * (b + copysignf(sqrt_disc, b));
    float t0 = q / a;
    float t1 = c / q;

    if (t0 > t1)
    {
        float tmp = t0;
        t0 = t1;
        t1 = tmp;
    }

    float t_candidate = 1e30f;

    if (t0 > EPSILON)
        t_candidate = t0;
    else if (t1 > EPSILON)
        t_candidate = t1;
    else
        return -1;

    float y = origin.y + t_candidate * dir.y;
    float half_len = reflector_len * 0.5f;
    if (y < -half_len || y > half_len)
        return -1;

    t += t_candidate;
    hit_point = add3(origin, mul3(dir, t_candidate));
    return 1;
}

/*
    @brief 计算光线在异介质的行为，并计算折射/反射方向
    @param I 入射方向
    @param N 法线方向
    @param n_i 入射介质折射率
    @param n_t 出射介质折射率
    @param Refract 折射/反射方向（单位向量）
    @param local_state 本地随机数状态，用于生成随机数
    @return 1 发生折射，2 发生菲涅尔反射,-1 发生全内反射
*/
__device__ int ray_refract(
    const vec3f &I,
    const vec3f &N,
    float n_i,
    float n_t,
    vec3f &Refract,
    curandState *local_state)
{
    float reflect_prob = fresnel(I, N, n_i, n_t);
    if (reflect_prob >= 1.0f)
    {
        Refract = normalize3(reflect3(I, N));
        return -1;
    } // 全反射
    if (curand_uniform(local_state) < reflect_prob)
    {
        Refract = normalize3(reflect3(I, N));
        return 2; // 菲涅尔反射
    }
    else
    {
        if (refract3(I, N, n_i, n_t, Refract))
            return 1; // 成功折射
    }
    return -1;
}

/*
    @brief 判断并计算射线与玻璃管内外表面相交，给出出射点和出射方向。
    @param origin 射线起点
    @param direction 射线方向
    @param leave_origin (如果有)反射/折射出射点
    @param leave_direction (如果有)反射/折射出射方向
    @param t 光线传播距离
    @param local_state 本地随机数状态，用于生成随机数
    @return 未命中返回-1，发生命中返回1
*/
__device__ int intersect_glass(
    const vec3f &origin,
    const vec3f &dir,
    vec3f &norigin,
    vec3f &ndir,
    float &energy_weight,
    float &t,
    curandState *local_state)
{
    vec3f hit_point_outer = {0.0f, 0.0f, 0.0f};
    vec3f hit_point_inner = {0.0f, 0.0f, 0.0f};
    vec3f N;
    vec3f center = {0.0f, 0.0f, receiver_h};
    float dist1 = 0.0f, dist2 = 0.0f;
    int hit_outer = ray_intersect_circle(origin, dir, center, glass_outer_R, dist1, hit_point_outer);
    int hit_inner = ray_intersect_circle(origin, dir, center, glass_inner_R, dist2, hit_point_inner);
    float n_i = 1.0f, n_t = Ior;
    int flag = -1;
    float dist_to_axis = sqrtf(origin.x * origin.x + (origin.z - receiver_h) * (origin.z - receiver_h));
    enum Region
    {
        OUTSIDE_AIR, // 玻璃管外部
        IN_GLASS,    // 玻璃管壁内部
        INSIDE_AIR   // 玻璃管内部空气中
    };
    Region current_region;
    if (dist_to_axis > glass_outer_R + EPSILON)
    {
        current_region = OUTSIDE_AIR;
        // printf("origin_to_axis_dist=%.6f, 外部空气命中\n", dist_to_axis);
    }
    else if (dist_to_axis >= glass_inner_R - EPSILON &&
             dist_to_axis <= glass_outer_R + EPSILON)
    {
        current_region = IN_GLASS;
        // printf("origin_to_axis_dist=%.6f, 玻璃管壁内命中\n", dist_to_axis);
    }
    else
    {
        current_region = INSIDE_AIR;
        // printf("origin_to_axis_dist=%.4f, 玻璃管内部命中\n", dist_to_axis);
    } // 判断当前光线处于何处

    if (hit_outer == -1 && hit_inner == -1)
    {
        norigin = origin;
        ndir = dir;
        return flag;
    }
    if (dist1 < EPSILON && dist2 < EPSILON)
    {
        // 没有有效的命中
        norigin = origin;
        ndir = dir;
        return flag;
    }

    // 选择正确的命中表面（基于当前位置）
    // 规则：根据光线行为，输出击中点（新起始点）以及新出射方向
    switch (current_region)
    {
    case OUTSIDE_AIR:
        // 从外部空气来，应该命中外表面
        if (hit_outer == 1 && dist1 > EPSILON)
        {
            t += dist1;
            N = normalize3({hit_point_outer.x, 0.0f, hit_point_outer.z - center.z});
            n_i = 1.0f;
            n_t = Ior;
            flag = 1; // outside -- glass外壁
            int refraction_result = ray_refract(dir, N, n_i, n_t, ndir, local_state);
            if (curand_uniform(local_state) < transmissivity)
            {
                if (refraction_result == -1)
                {
                    norigin = add3(hit_point_outer, mul3(N, SURFACE_EPSILON));
                    // printf("从外部空气来，命中外表面点(%.4f, %.4f, %.4f),全反射方向(%.4f, %.4f, %.4f),N=(%.4f,%.4f,%.4f,),norigin=(%.4f, %.4f, %.4f)\n",
                    //     hit_point_outer.x, hit_point_outer.y, hit_point_outer.z,
                    //     ndir.x, ndir.y, ndir.z, N.x, N.y, N.z, norigin.x, norigin.y, norigin.z);
                    energy_weight *= 1.0f;
                    return -1;
                }
                else if (refraction_result == 2)
                {
                    norigin = add3(hit_point_outer, mul3(N, SURFACE_EPSILON));
                    // printf("从外部空气来，命中外表面点(%.4f, %.4f, %.4f),全反射方向(%.4f, %.4f, %.4f),N=(%.4f,%.4f,%.4f,),norigin=(%.4f, %.4f, %.4f)\n",
                    //     hit_point_outer.x, hit_point_outer.y, hit_point_outer.z,
                    //     ndir.x, ndir.y, ndir.z, N.x, N.y, N.z, norigin.x, norigin.y, norigin.z);
                    energy_weight *= 1.0f;
                    return -1;
                }
                else
                {
                    norigin = sub3(hit_point_outer, mul3(N, 1.5f * SURFACE_EPSILON));
                    // printf("从外部空气来，命中外表面点(%.4f, %.4f, %.4f),", hit_point_outer.x, hit_point_outer.y, hit_point_outer.z);
                    // printf("折射方向(%.4f, %.4f, %.4f)\n", ndir.x, ndir.y, ndir.z);
                    // energy_weight = energy_weight*(1-fresnel(dir, N, n_i, n_t))* transmissivity;
                    energy_weight *= 1.0f;
                    return 1;
                }
            }
            else
            {
                energy_weight *= 0.0f;
                return -2;
            }
        }
        break;
    case IN_GLASS:
        // 在玻璃内部，可能射向外表面或内表面
        if (hit_outer == 1 && dist1 > EPSILON &&
            hit_inner == 1 && dist2 > EPSILON)
        {
            if (dist1 < dist2)
            {
                t += dist1;
                N = normalize3(sub3(hit_point_outer, {0.0f, hit_point_outer.y, glass_outer_h}));
                n_i = Ior;
                n_t = 1.0f;
                flag = 2; // 从内 -- glass外壁
                if (curand_uniform(local_state) < transmissivity)
                {
                    int refraction_result = ray_refract(dir, N, n_i, n_t, ndir, local_state);
                    if (refraction_result == -1)
                    {
                        // printf("从玻璃内来，命中外表面点(%.4f, %.4f, %.4f),", hit_point_outer.x, hit_point_outer.y, hit_point_outer.z);
                        // printf("全反射方向(%.4f, %.4f, %.4f)\n", ndir.x, ndir.y, ndir.z);
                        norigin = sub3(hit_point_outer, mul3(N, SURFACE_EPSILON));
                        energy_weight *= 1.0f;
                        return 2;
                    }
                    else if (refraction_result == 2)
                    {
                        norigin = sub3(hit_point_outer, mul3(N, SURFACE_EPSILON));
                        energy_weight *= 1.0f;
                        return 2;
                    }
                    else
                    {
                        // printf("从玻璃内来，命中外表面点(%.4f, %.4f, %.4f),", hit_point_outer.x, hit_point_outer.y, hit_point_outer.z);
                        // printf("折射方向(%.4f, %.4f, %.4f)\n", ndir.x, ndir.y, ndir.z);
                        norigin = add3(hit_point_outer, mul3(N, SURFACE_EPSILON));
                        // energy_weight = energy_weight*(1-fresnel(dir, N, n_i, n_t))* transmissivity;
                        energy_weight *= 1.0f;
                        return -1;
                    }
                }
                else
                {
                    energy_weight *= 0.0f;
                    return -2;
                }
            }
            else
            {
                t += dist2;
                N = normalize3(sub3(hit_point_inner, {0.0f, hit_point_inner.y, glass_inner_h}));
                n_i = Ior;
                n_t = 1.0f;
                flag = 3; // 从内 -- glass内壁
                int refraction_result = ray_refract(dir, N, n_i, n_t, ndir, local_state);
                if (refraction_result == -1)
                {
                    // printf("从玻璃内来，命中内表面点(%.4f, %.4f, %.4f),", hit_point_inner.x, hit_point_inner.y, hit_point_inner.z);
                    // printf("全反射方向(%.4f, %.4f, %.4f)\n", ndir.x, ndir.y, ndir.z);
                    norigin = add3(hit_point_inner, mul3(N, SURFACE_EPSILON));
                    energy_weight *= 1.0f;
                    return 3;
                }
                else if (refraction_result == 2)
                {
                    norigin = add3(hit_point_inner, mul3(N, SURFACE_EPSILON));
                    energy_weight *= 1.0f;
                    return 3;
                }
                else
                {
                    // printf("从玻璃内来，命中内表面点(%.4f, %.4f, %.4f),", hit_point_inner.x, hit_point_inner.y, hit_point_inner.z);
                    // printf("折射方向(%.4f, %.4f, %.4f)\n", ndir.x, ndir.y, ndir.z);
                    norigin = sub3(hit_point_inner, mul3(N, SURFACE_EPSILON));
                    // energy_weight = energy_weight*(1-fresnel(dir, N, n_i, n_t))* 1.0f;
                    energy_weight *= 1.0f;
                    return 4;
                }
            }
        }
        else if (hit_outer == 1 && dist1 > EPSILON)
        {
            t += dist1;
            N = normalize3(sub3(hit_point_outer, {0.0f, hit_point_outer.y, glass_outer_h}));
            n_i = Ior;
            n_t = 1.0f;
            flag = 2; // 从内 -- glass外壁
            if (curand_uniform(local_state) < transmissivity)
            {
                int refraction_result = ray_refract(dir, N, n_i, n_t, ndir, local_state);
                if (refraction_result == -1)
                {
                    // printf("从玻璃内来，命中外表面点(%.4f, %.4f, %.4f),", hit_point_outer.x, hit_point_outer.y, hit_point_outer.z);
                    // printf("全反射方向(%.4f, %.4f, %.4f)\n", ndir.x, ndir.y, ndir.z);
                    norigin = sub3(hit_point_outer, mul3(N, SURFACE_EPSILON));
                    energy_weight *= 1.0f;
                    return 2;
                }
                else if (refraction_result == 2)
                {
                    norigin = sub3(hit_point_outer, mul3(N, SURFACE_EPSILON));
                    energy_weight *= 1.0f;
                    return 2;
                }
                else
                {
                    // printf("从玻璃内来，命中外表面点(%.4f, %.4f, %.4f),", hit_point_outer.x, hit_point_outer.y, hit_point_outer.z);
                    // printf("折射方向(%.4f, %.4f, %.4f)\n", ndir.x, ndir.y, ndir.z);
                    norigin = add3(hit_point_outer, mul3(N, SURFACE_EPSILON));
                    // energy_weight = energy_weight*(1-fresnel(dir, N, n_i, n_t))* transmissivity;
                    energy_weight *= 1.0f;
                    return -1;
                }
            }
            else
            {
                energy_weight *= 0.0f;
                return -2;
            }
        }
        else if (hit_inner == 1 && dist2 > EPSILON)
        {
            t += dist2;
            N = normalize3(sub3(hit_point_inner, {0.0f, hit_point_inner.y, glass_inner_h}));
            n_i = Ior;
            n_t = 1.0f;
            flag = 3; // 从内 -- glass内壁
            int refraction_result = ray_refract(dir, N, n_i, n_t, ndir, local_state);
            if (refraction_result == -1)
            {
                // printf("从玻璃内来，命中内表面点(%.4f, %.4f, %.4f),", hit_point_inner.x, hit_point_inner.y, hit_point_inner.z);
                // printf("全反射方向(%.4f, %.4f, %.4f)\n", ndir.x, ndir.y, ndir.z);
                norigin = add3(hit_point_inner, mul3(N, SURFACE_EPSILON));
                energy_weight *= 1.0f;
                return 3;
            }
            else if (refraction_result == 2)
            {
                norigin = add3(hit_point_outer, mul3(N, SURFACE_EPSILON));
                energy_weight *= 1.0f;
                return 3;
            }
            else
            {
                // printf("从玻璃内来，命中内表面点(%.4f, %.4f, %.4f),", hit_point_inner.x, hit_point_inner.y, hit_point_inner.z);
                // printf("折射方向(%.4f, %.4f, %.4f)\n", ndir.x, ndir.y, ndir.z);
                norigin = sub3(hit_point_inner, mul3(N, SURFACE_EPSILON));
                // energy_weight = energy_weight*(1-fresnel(dir, N, n_i, n_t))* 1.0f;
                energy_weight *= 1.0f;
                return 4;
            }
        }
        break;
    case INSIDE_AIR:
        // 从内部空气来，应该命中内表面
        if (hit_inner == 1 && dist2 > EPSILON)
        {
            t += dist2;
            N = normalize3(sub3(hit_point_inner, {0.0f, hit_point_inner.y, glass_inner_h}));
            n_i = 1.0f;
            n_t = Ior;
            flag = 4; // inside -- glass内壁
            int refraction_result = ray_refract(dir, mul3(N, -1.0f), n_i, n_t, ndir, local_state);
            if (refraction_result == -1)
            {
                // printf("从内部空气来，命中内表面点(%.4f, %.4f, %.4f),", hit_point_inner.x, hit_point_inner.y, hit_point_inner.z);
                // printf("全反射方向(%.4f, %.4f, %.4f)\n", ndir.x, ndir.y, ndir.z);
                norigin = sub3(hit_point_inner, mul3(N, SURFACE_EPSILON));
                energy_weight *= 1.0f;
                return 4;
            }
            else if (refraction_result == 2)
            {
                norigin = sub3(hit_point_inner, mul3(N, SURFACE_EPSILON));
                energy_weight *= 1.0f;
                return 4;
            }
            else
            {
                // printf("从内部空气来，命中内表面点(%.4f, %.4f, %.4f),", hit_point_inner.x, hit_point_inner.y, hit_point_inner.z);
                // printf("折射方向(%.4f, %.4f, %.4f)\n", ndir.x, ndir.y, ndir.z);
                norigin = add3(hit_point_inner, mul3(N, SURFACE_EPSILON));
                // energy_weight = energy_weight*(1-fresnel(dir, N, n_i, n_t))* transmissivity;
                energy_weight *= 1.0f;
                return 3;
            }
        }
        break;
    }
    return -1;
}

// --------------------吸收管部分--------------------
/*
    @brief 检查射线是否与吸收管表面相交;留了集热管表面发生反射的接口，但暂不实现
    @param origin 射线起点
    @param direction 射线方向
    @param t 光线传播距离
    @return 命中的吸收管分段索引(0 ~ TOTAL_SEGMENT-1)，未命中返回-1
*/
__device__ int intersect_outer_absorber(
    const vec3f &origin,
    const vec3f &direction,
    float &t,
    vec3f &hit_Point)
{
    int hit_segment = -2;
    vec3f hit_point = {0.0f, 0.0f, 0.0f};
    float dist = 0.0f;
    int hit = ray_intersect_circle(origin, direction, {0.0f, 0.0f, receiver_h}, receiver_R, dist, hit_point);
    if (hit == -1)
    {
        return -1;
    }
    else if (hit == 1)
    {
        t += dist;
        // 光线被吸收
        vec3f centralP = {hit_point.x, 0.0f, hit_point.z - receiver_h};
        float hit_theta = atan2f(centralP.z, centralP.x) - 0.5 * PI;
        if (hit_theta < 0.0f)
            hit_theta += 2.0f * (float)PI;

        float theta_step = 2.0f * (float)PI / (float)M;
        int theta_segment = std::floor((float)hit_theta / (float)theta_step);
        if (theta_segment >= M)
            theta_segment -= M;
        float y_step = receiver_len / (float)N;
        int y_segment = std::floor((hit_point.y + receiver_len * 0.5f) / y_step);
        hit_segment = theta_segment + M * y_segment;
        hit_Point = hit_point;
    }
    else
    {
        hit_segment = -1;
    }
    return hit_segment;
}

/*
    @brief device：判断射线是否与平行四边形相交
    @param origin 射线起点
    @param direction 射线方向
    @param A 平行四边形的一个顶点
    @param B 平行四边形的另一个顶点
    @param C 平行四边形的第三个顶点
    @return 是否相交
*/
__device__ bool ray_intersects_Parallelogram(
    const vec3f &origin, const vec3f &direction,
    const vec3f &normal, const vec3f &A, const vec3f &B, const vec3f &C)
{
    vec3f dir = normalize3(direction);
    vec3f v1 = sub3(B, A);
    vec3f v2 = sub3(C, A);
    vec3f n = normal;
    float denom = dot3(dir, n);
    vec3f P = {0.0f, 0.0f, 0.0f};
    float t = 0.0f;
    if (fabs(denom) > EPSILON)
    {
        t = dot3(sub3(A, origin), n) / denom;
        if (t >= 0)
        {
            P = add3(origin, mul3(dir, t));
            vec3f AP = sub3(P, A);
            float d1 = dot3(AP, v1);
            float d2 = dot3(AP, v2);
            if ((0 <= d1) && (d1 <= dot3(v1, v1)) && (0 <= d2) && (d2 <= dot3(v2, v2)))
            {
                return true;
            }
        }
    }
    return false;
}

/*
    @brief device：使用ray_intersects_Parallelogram法判断射线是否与接收平面相交
    @param origin 射线起点
    @param direction 射线方向
    @return 是否相交
*/
__device__ bool is_blocked(const vec3f &origin, const vec3f &direction)
{
    vec3f tmp_A = {-plane_width * 0.5f, plane_len * 0.5f, plane_z};
    vec3f tmp_B = {plane_width * 0.5f, plane_len * 0.5f, plane_z};
    vec3f tmp_C = {plane_width * 0.5f, -plane_len * 0.5f, plane_z};
    vec3f normal = {0.0f, 0.0f, 1.0f};
    if (ray_intersects_Parallelogram(origin, direction, normal, tmp_A, tmp_B, tmp_C))
    {
        return true;
    }
    return false;
}

// --------------------Kernel部分--------------------
/*
    @brief 生成射线并追踪的主内核
    @param states 随机数状态数组
    @param hits 命中结果数组
    @param bounce_list  二次反射镜弹射次数数组
    @param bounce_distribution 碰撞次数分布数组
    @param nRays 射线数量
    @param ppr power_per_ray 每条射线的能量
    @param slope_error 斜率误差
*/
__global__ void generate_ray_and_trace_CPC_kernel(
    curandState *states,
    CyliReflectors *d_reflectors_ptr,
    Sun *d_sun_ptr,
    float *energy_absorbed,
    int *bounce_list,
    int nRays,
    float ppr,
    float slope_error)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= nRays)
        return;
    curandState local_state = states[idx];

    // 计算当前光线对应的一次反射镜及其上的像素位置
    int refID = idx / (Lpixel * Wpixel * reflector_spp);                     // 一次镜ID
    int pixelID = (idx % (Lpixel * Wpixel * reflector_spp)) / reflector_spp; // 当前光线在一次镜上的ID
    vec3f coor_origin = d_reflectors_ptr->sample_on_CyliRefs(refID, pixelID, &local_state);
    vec3f nor = d_reflectors_ptr->getRefNormal(refID);
    vec3f Normal_atP = rotate3(d_reflectors_ptr->getRefNormalatPoint(refID, coor_origin), nor); // 当前一次反射镜在采样点的设定法向
    vec3f origin = rotate3(coor_origin, nor);
    origin = add3(origin, d_reflectors_ptr->getRefPosition(refID));
    Normal_atP = perturb_N(Normal_atP, &local_state, slope_error);        // 斜率误差扰动后的镜面法向
    vec3f idir = d_sun_ptr->SampleGaussianSunshape(&local_state, 0.004f); // 该像素一次反射镜的入射方向
    idir = normalize3(idir);

    // 检测由二次反射镜造成的遮挡，这里按平面遮挡简化处理。idir要做负向处理，因为原始设定为太阳 --> 反射镜
    if (is_blocked(origin, mul3(idir, -1)))
    {
        states[idx] = local_state;
        return;
    }
    // 计算反射方向
    vec3f rdir = normalize3(reflect3(idir, Normal_atP)), dir = rdir;

    // 检测出射光线与平面的相交，如果不相交直接退出
    if (fabs(rdir.z) < EPSILON)
    {
        states[idx] = local_state;
        return;
    }
    float t = (plane_z - origin.z) / rdir.z;
    if (t <= EPSILON)
    {
        states[idx] = local_state;
        return;
    }
    vec3f hit_plane_point = add3(origin, mul3(rdir, t));
    if (hit_plane_point.x > plane_width * 0.5f || hit_plane_point.x < -plane_width * 0.5f || hit_plane_point.y > plane_len * 0.5f || hit_plane_point.y < -plane_len * 0.5f)
    {
        states[idx] = local_state;
        return;
    }

    int count = 0;
    float energy_weight = 1.0f;
    int flag = 0;
    t = 0.0f; // 起点重置为一次镜表面
    // float cosinc = 0;
    float cosinc = dot3(rdir, Normal_atP); // 当前光线的反射方向与反射镜微元法向之间的夹角余弦，代表其在光线反射方向上的有效投影面积
    /*
        flag说明：
        -1: 从外壁向外射出/未命中玻璃管系统
        1：从玻璃外壁折射入玻璃环
        2：从玻璃环内在玻璃外壁发生反射
        3: 从玻璃环内在玻璃内壁发生反射
        4: 从玻璃内壁折射入玻璃管内部空气
    */
    while (count < 10)
    {
        vec3f norigin, ndir;
        if (energy_weight < 1e-6f)
        {
            break; // 提前终止这条光线
        }
        flag = intersect_glass(origin, dir, norigin, ndir, energy_weight, t, &local_state);
        // printf("idx=%d,refID=%d,flag=%d,count=%d,o=(%.4f, %.4f, %.4f),d=(%.4f, %.4f, %.4f),no=(%.4f, %.4f, %.4f),nd=(%.4f, %.4f, %.4f),t=%.2f\n",
        // idx,refID,flag,count,origin.x,origin.y,origin.z,dir.x,dir.y,dir.z,norigin.x,norigin.y,norigin.z,ndir.x,ndir.y,ndir.z,t);
        if (flag == -2)
            break;
        if (flag == 1 || flag == 2 || flag == 3)
        {
            origin = norigin;
            dir = ndir;
            count++;
        }
        else if (flag == 4)
        {
            vec3f hit_point;
            count++;
            int segment = intersect_outer_absorber(norigin, ndir, t, hit_point);
            if (segment != -1)
            {
                count++;
                float eta = 0.99321 - 0.0001176 * t + 1.97 * 1e-8 * t * t;
                vec3f tmp_N = normalize3(sub3(hit_point, {0.0f, hit_point.y, receiver_h}));
                // ndir = normalize3(ndir);
                // cosinc = fabs(dot3(normalize3({ndir.x,0.0f,ndir.z}), tmp_N));
                atomicAdd(&energy_absorbed[segment], energy_weight * ppr * cosinc * eta * 0.92);
                // printf("idx=%d,count=%d,o=(%.4f, %.4f, %.4f),d=(%.4f, %.4f, %.4f),no=(%.4f, %.4f, %.4f),nd=(%.4f, %.4f, %.4f),t=%.2f,eta=%.4f,flux=%.2f\n",
                //        idx, count, origin.x, origin.y, origin.z, dir.x, dir.y, dir.z, norigin.x, norigin.y, norigin.z, ndir.x, ndir.y, ndir.z, t, eta, energy_weight * ppr * cosinc * eta);
                // printf("idx=%d,refID=%d,count=%d,no=(%.4f, %.4f, %.4f),nd=(%.4f, %.4f, %.4f),p=(%.4f, %.4f, %.4f),N_at_p=(%.4f, %.4f, %.4f),cosinc=%.4f,t=%.2f,flux=%.2f\n",
                    //    idx, refID,count,norigin.x, norigin.y, norigin.z, ndir.x, ndir.y, ndir.z, hit_point.x, hit_point.y, hit_point.z, tmp_N.x, tmp_N.y, tmp_N.z, cosinc, t, energy_weight * ppr * cosinc * eta);
                bounce_list[idx] = count;
                break;
            }
            else
            {
                origin = norigin;
                dir = ndir;
                count++;
            }
        }
        else
        {
            // if (!intersect_secondary_reflector(origin, dir, norigin, ndir, t,&local_state,slope_error))
            // break; // 玻璃管、二次镜皆未命中
            if (!intersect_secondary_parabola(origin, dir, norigin, ndir, t, &local_state, slope_error, refID, count))
                break; // 玻璃管、二次镜皆未命中
            if (curand_uniform(&local_state) >= specular_reflectivity)
                break; // 命中，但被二次镜吸收
            // 光线与未击中玻璃管系统，而与二次反射镜相交并反射
            origin = norigin;
            dir = ndir;
            energy_weight *= specular_reflectivity;
            count++;
        }
    }
    states[idx] = local_state;
}

void SaveAsCSV(thrust::host_vector<float> h_fluxdata, std::string filename)
{
    std::ofstream file(filename, std::ios::out);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }
    if (file.tellp() == 0)
        file << "seg_th,part_energy(W/m2)\n";
    for (int i = 0; i < TOTAL_SEGMENTS; ++i)
    {
        file << std::fixed << std::setprecision(6) << i << "," << h_fluxdata[i] << "\n";
        if ((i + 1) % M == 0 && i != 0)
            file << "\n";
    }

    file.close();
    std::cout << "Data successfully saved to " << filename << std::endl;
}

int main()
{
    const float DNI = 1000.0f;
    vec3f sun_dir = {0.0f, 0.0f, -1.0f};
    const int nRays = reflector_num * Lpixel * Wpixel * reflector_spp;
    // 初始化CUDA设备
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << std::endl;
    float S_hsub = (reflector_len / Lpixel) * (reflector_width / Wpixel);
    float S_rsub = (receiver_len / N) * (2 * PI * receiver_R / M);
    float power_per_ray = DNI * S_hsub / (reflector_spp * S_rsub); // 每条射线的能量
    printf("power_per_ray=%.6f(W/m2)\n", power_per_ray);
    // 初始化一次曲面反射镜
    CyliReflectors Reflectors(reflector_num, sun_dir);
    CyliReflectors *d_reflectors_ptr = Reflectors.createDeviceCopy();
    // Reflectors.display_DEBUG_info();
    Sun h_sun({0.0f, 0.0f, 100.0f}, 100.0f, GAUSSIAN);
    Sun *d_sun = nullptr;

    // --------------------初始化随机状态--------------------
    curandState *d_states = nullptr;
    CHECK(cudaMalloc(&d_states, nRays * sizeof(curandState)));
    curandState *d_states_pool = nullptr;
    CHECK(cudaMalloc(&d_states_pool, RAND_STATE_POOL_SIZE * sizeof(curandState)));
    auto start = std::chrono::high_resolution_clock::now();
    setup_kernel_pool<<<(RAND_STATE_POOL_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states_pool, (unsigned long)time(nullptr), RAND_STATE_POOL_SIZE); // 设定池子
    assign_random_states<<<(nRays + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states_pool, d_states, nRays, RAND_STATE_POOL_SIZE);                          // 分配随机状态
    CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Random states initialization complete. Run time: " << elapsed.count() * 1000 << " ms" << std::endl;
    // --------------------初始化随机状态--------------------

    // --------------------主函数变量区--------------------
    thrust::device_vector<float> d_energy_absorbed(TOTAL_SEGMENTS, 0.0f);
    thrust::device_vector<float> d_energy_absorbed_filtered(TOTAL_SEGMENTS, 0.0f);
    thrust::device_vector<int> d_bounce_list(nRays, 0);
    dim3 block(BLOCK_SIZE);
    dim3 grid((nRays + block.x - 1) / block.x);
    thrust::host_vector<float> h_energy_absorbed(TOTAL_SEGMENTS, 0.0f);
    thrust::host_vector<float> h_energy_absorbed_filtered(TOTAL_SEGMENTS, 0.0f);
    double avg_time[ROUND_TIME] = {0.0};
    double time_avg = 0.0;
    std::vector<float> total_energy(ROUND_TIME, 0.0f), total_energy_filtered(ROUND_TIME, 0.0f), total_hit(ROUND_TIME, 0.0f);
    float energy_avg = 0.0f, energy_avg_filtered = 0.0f, hit_avg = 0.0f;
#ifdef BOUNCE_TEST
    std::ofstream fout_bounce("bounce_distribution.csv", std::ios::app);
#endif

    // --------------------主函数变量区--------------------

#ifdef BOUNCE_TEST
    if (fout_bounce.tellp() == 0)
    {
        // 碰撞分布文件表头：斜率误差 + 各碰撞次数的计数
        fout_bounce << "slope_error(mrad),total_rays";
        for (int bounce = 1; bounce < MAX_BOUNCES; bounce++)
        {
            fout_bounce << ",bounce_" << bounce << "_count";
        }
        fout_bounce << "\n";
    }
#endif
    // --------------------数据文件创建--------------------

    // --------------------主程序--------------------
    std::cout << "LFR Flux Simulation by RMCRT begins. Total rays: " << nRays << std::endl;
    std::cout << "Slope error of Reflector: " << slope_error[1] << "rad " << std::endl; // slope error = 2.5mrad
    std::cout << "Use Sunshape: ";
    if (h_sun.GetSunshape() == GAUSSIAN)
        std::cout << "Gaussian Sunshape" << std::endl;
    else if (h_sun.GetSunshape() == BUIE)
        std::cout << "Buie Sunshape" << std::endl;
    else
        std::cout << "Uniform Sunshape" << std::endl;

    float heightAngle = 90.0;
    float azimuthAngle = 90.0;
    printf("Time: %d h, Height: %.2f°, Azimuth: %.2f°\n", 12, heightAngle, azimuthAngle);
    h_sun.ConvertToDirection(heightAngle, azimuthAngle);
    h_sun.display_DEBUG_info();

    // float lp0x = 0.0f, lp0z = p0z; // 8.14
    // float lp1x = -0.05f, lp1z = p1z; // 8.1109f
    // float lp2x = -0.15f, lp2z = p2z; // 7.8784f
    // float rp0x = 0.0f, rp0z = p0z;
    // float rp1x = 0.05f, rp1z = p1z;
    // float rp2x = 0.15f, rp2z = p2z;
    // for(float t=0;t<=1.0f;t+=0.05){
    //     float lx = bezier_curve(lp0x, lp1x, lp2x, t);
    //     float lz = bezier_curve(lp0z, lp1z, lp2z, t);
    //     float rx = bezier_curve(rp0x, rp1x, rp2x, t);
    //     float rz = bezier_curve(rp0z, rp1z, rp2z, t);
    //     printf("t=%.2f, lx=%.2f, lz=%.4f, rx=%.2f, rz=%.4f, z_soltrace=%.4f\n", t, lx, lz, rx, rz, 8.14-lx*lx/0.086);
    // }

    for (int round = 0; round < ROUND_TIME; round++)
    {
        // std::cout << "Round " << round + 1 << " begins. " << std::endl;
        thrust::fill(d_bounce_list.begin(), d_bounce_list.end(), 0);
        thrust::fill(d_energy_absorbed.begin(), d_energy_absorbed.end(), 0.0f);

        d_sun = h_sun.CreateDeviceCopy();
        // Kernel主函数
        start = std::chrono::high_resolution_clock::now();
        generate_ray_and_trace_CPC_kernel<<<grid, block>>>(
            d_states,
            d_reflectors_ptr,
            d_sun,
            thrust::raw_pointer_cast(d_energy_absorbed.data()),
            thrust::raw_pointer_cast(d_bounce_list.data()),
            nRays,
            power_per_ray,
            slope_error[1]);
        CHECK(cudaDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        // std::cout << "Round " << round + 1 << " " << "kernel execution complete. Run time: " << elapsed.count() * 1000 << " ms" << std::endl;
        avg_time[round] = elapsed.count() * 1000;
        std::vector<int> h_bounces(nRays);
        thrust::copy(d_bounce_list.begin(), d_bounce_list.end(), h_bounces.begin());
        // std::cout << "Round " << round + 1 << " bounce distribution:" << std::endl;
        // 统计碰撞分布
        std::map<int, int> hist;
        for (int b : h_bounces)
        {
            if (b > 0)
                hist[b]++; // 统计击中的
        }

        // std::cout << "Bounce distribution (count of rays hitting tube):\n";
        int totalhit = 0;
        for (auto &kv : hist)
        {
            // std::cout << "  " << kv.first << " bounces: " << kv.second << std::endl;
            totalhit += kv.second;
        }
        total_hit[round] = totalhit;

        h_energy_absorbed = d_energy_absorbed;

        int grid_width = M;  // 角度方向分段数
        int grid_height = N; // 长度方向分段数

        // apply_tms_filter(
        //     thrust::raw_pointer_cast(d_energy_absorbed.data()),
        //     thrust::raw_pointer_cast(d_energy_absorbed_filtered.data()),
        //     grid_width,
        //     grid_height
        // );
        apply_gaussian_filter(
            thrust::raw_pointer_cast(d_energy_absorbed.data()),
            thrust::raw_pointer_cast(d_energy_absorbed_filtered.data()),
            grid_width,
            grid_height,
            3.0f // sigma值，可调节
        );
        CHECK(cudaDeviceSynchronize());
        // 拷贝到主机内存
        h_energy_absorbed_filtered = d_energy_absorbed_filtered;

        // 结果计算，原来+滤波
        for (int i = 0; i < TOTAL_SEGMENTS; i++)
        {
            // 原始结果
            total_energy[round] += h_energy_absorbed[i];
            // 滤波结果
            total_energy_filtered[round] += h_energy_absorbed_filtered[i];
        }
#ifdef BOUNCE_TEST
        // 输出碰撞分布统计
        std::cout << "Round " << round + 1 << " bounce distribution:" << std::endl;
        int total_counted_rays = 0;
        for (int i = 0; i < MAX_BOUNCE_CATEGORIES; i++)
        {
            if (h_bounce_distribution[i] > 0)
            {
                float percentage = (float)h_bounce_distribution[i] / nRays * 100.0f;
                std::cout << "  " << i << " bounces: " << h_bounce_distribution[i]
                          << " rays (" << percentage << "%)" << std::endl;
                total_counted_rays += h_bounce_distribution[i];
            }
        }

        // 检查是否有光线未被统计（理论上应该等于nRays）
        if (total_counted_rays != nRays)
        {
            std::cout << "Warning: " << nRays - total_counted_rays
                      << " rays not accounted for in bounce distribution!" << std::endl;
        }

        std::vector<int> h_bounces(nRays);
        thrust::copy(d_bounce_list.begin(), d_bounce_list.end(), h_bounces.begin());
        std::cout << "Round " << round + 1 << " bounce distribution:" << std::endl;
        // 统计碰撞分布
        std::map<int, int> hist;
        for (int b : h_bounces)
        {
            if (b >= 0)
                hist[b]++; // 全部统计
        }

        std::cout << "Bounce distribution (count of rays hitting tube):\n";
        for (auto &kv : hist)
        {
            // std::cout << "  " << kv.first << " bounces: " << kv.second << std::endl;
        }
        // 写入碰撞分布到CSV
        fout_bounce << std::fixed << std::setprecision(6)
                    << slope_error[1] * 1000.0f << ","
                    << nRays;

        for (int bounce = 1; bounce < MAX_BOUNCES; bounce++)
        {
            fout_bounce << "," << hist[bounce];
        }
        fout_bounce << "\n";
#endif
    }
    // --------------------数据统计与文件写入--------------------

    for (int round = 0; round < ROUND_TIME; round++)
    {
        time_avg += avg_time[round];
        energy_avg += total_energy[round];
        energy_avg_filtered += total_energy_filtered[round];
        hit_avg += total_hit[round];
    }
    hit_avg /= (float)ROUND_TIME;
    energy_avg /= (float)ROUND_TIME;
    energy_avg_filtered /= (float)ROUND_TIME;
    time_avg /= (float)ROUND_TIME;
    std::cout << "Average Kernel Execute Time: " << time_avg << " ms" << std::endl;
    std::cout << "Total ray hit : " << hit_avg << "(" << hit_avg / nRays << ")" << std::endl;
    std::cout << "Total energy - Origin: " << energy_avg << std::endl;
    std::cout << "Total energy - Filtered: " << energy_avg_filtered << std::endl;
    std::cout << "Energy ratio - Filtered / Origin: " << energy_avg_filtered / energy_avg << std::endl;
    SaveAsCSV(h_energy_absorbed, "CUDA_MCRT_energy_absorbed_origin.csv");
    // SaveAsCSV(h_energy_absorbed, "CUDA_MCRT_energy_absorbed_origin_noCPC.csv");
    SaveAsCSV(h_energy_absorbed_filtered, "CUDA_MCRT_energy_absorbed_filtered.csv");
    // SaveAsCSV(h_energy_absorbed_filtered, "CUDA_MCRT_energy_absorbed_filtered_noCPC.csv");
    // SaveAsCSV(h_energy_absorbed_filtered, "CUDA_MCRT_energy_absorbed_filtered_noGlass.csv");
    // SaveAsCSV(h_energy_absorbed_filtered, "CUDA_MCRT_energy_Aperture_filtered.csv");
#ifdef BOUNCE_TEST
    fout_bounce.close();
#endif
    // --------------------数据统计与文件写入--------------------

    // 释放内存
    CHECK(cudaFree(d_states_pool));
    CHECK(cudaFree(d_states));
    CHECK(cudaFree(d_reflectors_ptr));
    return 0;
}