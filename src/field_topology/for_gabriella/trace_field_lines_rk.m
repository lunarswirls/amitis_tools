function traj_all = trace_field_lines_rk(x_seed, y_seed, z_seed, Vx, Vy, Vz, ...
    x_grid, y_grid, z_grid, max_steps)
%
% Trace multiple field lines using fixed-step Runge-Kutta-Fehlberg 4(5)
% integration from seed points. Traces terminate when leaving the grid bounds
% or reaching max_steps. No surface termination.
%
% Inputs:
%   x_seed, y_seed, z_seed - arrays of starting positions (Nx1)
%   Vx, Vy, Vz             - 3D vector field components (Nx x Ny x Nz)
%   x_grid, y_grid, z_grid - 1D coordinate grids (monotonic increasing)
%   max_steps              - maximum integration steps per line (default: 5000)
%
% Outputs:
%   traj_all - cell array {N} of trajectories, each (T x 3) where T <= max_steps
%
% Notes:
%   - Grids assumed monotonic increasing; fields on meshgrid format
%   - Uses built-in interp3 equivalent via manual trilinear interp
%   - Step size auto-chosen as 1/2 of min grid spacing
%   - Fixed-step RK45 (5th order accurate)

    if nargin < 10
        max_steps = 5000;
    end

    % Ensure column vectors
    x_seed = x_seed(:);
    y_seed = y_seed(:);
    z_seed = z_seed(:);
    nSeeds = numel(x_seed);

    % Auto-select step size based on grid resolution
    h = choose_step_size(x_grid, y_grid, z_grid);

    % Trace each seed independently
    traj_all = cell(nSeeds, 1);
    for n = 1:nSeeds
        seed = [x_seed(n); y_seed(n); z_seed(n)];
        traj_all{n} = trace_single_line(seed, Vx, Vy, Vz, ...
            x_grid, y_grid, z_grid, max_steps, h);
    end
end


function h = choose_step_size(x_grid, y_grid, z_grid)
%
% Heuristic step size selector. Uses 1/2 of the smallest grid spacing across
% all dimensions to ensure ~2 points per cell on average.
%
% Inputs:
%   x_grid, y_grid, z_grid - 1D grids
%
% Output:
%   h - suggested fixed step size

    dx = min(diff(x_grid));
    dy = min(diff(y_grid));
    dz = min(diff(z_grid));
    h = 0.5 * min([dx, dy, dz]);
end


function traj = trace_single_line(seed, Vx, Vy, Vz, x_grid, y_grid, z_grid, ...
    max_steps, h)
%
% Trace one field line from seed using RK45 integration.
%
% Termination conditions (in order checked):
%   1. Vector field magnitude == 0
%   2. Position leaves x or z grid bounds
%   3. Position leaves y grid bounds
%   4. max_steps reached
%
% Inputs:
%   seed        - 3x1 starting position
%   Vx/Vy/Vz    - vector field on grids
%   x_grid/...  - coordinate grids
%   max_steps   - max steps
%   h           - fixed step size
%
% Output:
%   traj - (T x 3) trajectory positions

    traj = nan(max_steps, 3);
    traj(1, :) = seed(:).';

    r = seed(:);  % current position (column vector)

    for i = 2:max_steps
        % Get normalized unit vector field at current position
        V = get_unit_vector(r, Vx, Vy, Vz, x_grid, y_grid, z_grid);

        % Early termination: null field
        if all(V == 0)
            traj = traj(1:i-1, :);
            return;
        end

        % Single RK45 integration step
        r = rk45_step(@get_unit_vector, r, h, Vx, Vy, Vz, x_grid, y_grid, z_grid);

        % Store new position
        traj(i, :) = r(:).';

        % Termination: left x or z bounds (prioritized over y)
        if r(1) < x_grid(1) || r(1) > x_grid(end) || ...
           r(3) < z_grid(1) || r(3) > z_grid(end)
            traj = traj(1:i, :);
            return;
        end

        % Termination: left y bounds
        if r(2) < y_grid(1) || r(2) > y_grid(end)
            traj = traj(1:i, :);
            return;
        end
    end

    % Reached max_steps
    traj = traj(1:max_steps, :);
end


function v_unit = get_unit_vector(r, Vx, Vy, Vz, x_grid, y_grid, z_grid)
%
% Interpolate vector field at position r, return normalized unit vector.
% Zero vector if magnitude is zero or interpolation fails.
%
% Inputs:
%   r     - 3x1 position
%   Vx/...- field components
%   grids - coordinate arrays
%
% Output:
%   v_unit - 3x1 unit vector

    vx = trilinear_interp(x_grid, y_grid, z_grid, Vx, r(1), r(2), r(3));
    vy = trilinear_interp(x_grid, y_grid, z_grid, Vy, r(1), r(2), r(3));
    vz = trilinear_interp(x_grid, y_grid, z_grid, Vz, r(1), r(2), r(3));

    v = [vx; vy; vz];
    nrm = norm(v);

    if nrm == 0
        v_unit = zeros(3, 1);
    else
        v_unit = v / nrm;
    end
end


function val = trilinear_interp(x_grid, y_grid, z_grid, V, xi, yi, zi)
%
% Manual trilinear interpolation matching NumPy's griddata behavior.
% Clamps indices to valid range [1, size(V)-1]. Linearly extrapolates if
% outside but clamped.
%
% Inputs:
%   x_grid/... - 1D monotonic increasing coordinates
%   V          - (Nx x Ny x Nz) field values
%   xi,yi,zi   - query point
%
% Output:
%   val - interpolated scalar

    % Find lower index using search (equivalent to searchsorted-1)
    i = find(x_grid <= xi, 1, 'last');
    j = find(y_grid <= yi, 1, 'last');
    k = find(z_grid <= zi, 1, 'last');

    % Clamp to valid cell range
    i = max(1, min(i, numel(x_grid)-1));
    j = max(1, min(j, numel(y_grid)-1));
    k = max(1, min(k, numel(z_grid)-1));

    % Normalized distances within cell [0,1]
    xd = (xi - x_grid(i)) / (x_grid(i+1) - x_grid(i));
    yd = (yi - y_grid(j)) / (y_grid(j+1) - y_grid(j));
    zd = (zi - z_grid(k)) / (z_grid(k+1) - z_grid(k));

    % 8 corner values
    c000 = V(i,   j,   k);
    c100 = V(i+1, j,   k);
    c010 = V(i,   j+1, k);
    c001 = V(i,   j,   k+1);
    c101 = V(i+1, j,   k+1);
    c011 = V(i,   j+1, k+1);
    c110 = V(i+1, j+1, k);
    c111 = V(i+1, j+1, k+1);

    % Bilinear in x-y planes
    c00 = c000*(1-xd) + c100*xd;
    c01 = c001*(1-xd) + c101*xd;
    c10 = c010*(1-xd) + c110*xd;
    c11 = c011*(1-xd) + c111*xd;

    % Linear in y
    c0 = c00*(1-yd) + c10*yd;
    c1 = c01*(1-yd) + c11*yd;

    % Linear in z
    val = c0*(1-zd) + c1*zd;
end


function r_next = rk45_step(f, r, h, Vx, Vy, Vz, x_grid, y_grid, z_grid)
%
% Single fixed-step Runge-Kutta-Fehlberg 4(5) step (5th-order accurate).
% Uses classical RKF45 Butcher tableau coefficients for the final advance.
% No error estimation/adaptive stepping.
%
% Inputs:
%   f     - function handle returning unit vector dr/ds
%   r     - current 3x1 position
%   h     - step size along field line (arc length)
%   Vx/...- field data for f
%
% Output:
%   r_next - 3x1 position after one step

    k1 = f(r, Vx, Vy, Vz, x_grid, y_grid, z_grid);

    k2 = f(r + h * 0.25               * k1, ...
           Vx, Vy, Vz, x_grid, y_grid, z_grid);

    k3 = f(r + h * (3*k1  + 9*k2 ) / 32, ...
           Vx, Vy, Vz, x_grid, y_grid, z_grid);

    k4 = f(r + h * (1932*k1 -7200*k2 +7296*k3)/2197, ...
           Vx, Vy, Vz, x_grid, y_grid, z_grid);

    k5 = f(r + h * ( 439*k1/216 -8*k2     +3680*k3/513 -845*k4/4104), ...
           Vx, Vy, Vz, x_grid, y_grid, z_grid);

    k6 = f(r + h * (  -8*k1/27 +2*k2 -3544*k3/2565 +1859*k4/4104 -11*k5/40), ...
           Vx, Vy, Vz, x_grid, y_grid, z_grid);

    % 5th-order weights for position advance
    r_next = r + h * (16*k1/135    + 6656*k3/12825 + ...
                      28561*k4/56430 - 9*k5/50     + 2*k6/55);
end
