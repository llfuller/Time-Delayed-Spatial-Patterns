function sol = Oscillators2D
% Using DDE by The MathWorks, Inc.
% See DDE tutorial pdf.

% Import values
W = readNPY(['generated_values/W.npy']);
%dist_grid_arr = readNPY(['generated_values/dist_grid_arr.npy']);
%phi = readNPY(['generated_values/phi.npy']);
N = readNPY(['generated_values/N.npy']);
initial_conditions = readNPY(['generated_values/initial_conditions.npy']);
scalars = readNPY(['generated_values/scalars.npy']);
lag_times_reduced = readNPY(['generated_values/lags_reduced.npy']);
lag_indices = readNPY(['generated_values/lag_indices.npy']);
w = scalars(1);
K = scalars(2);
%v = scalars(3);
%Omega = scalars(4);
%gamma = scalars(5);
num_rows = scalars(6);
num_cols = scalars(7);
%numTimeSteps = scalars(8);
size_of_lag_matrix = size(lag_times_reduced);
num_lag_values = size_of_lag_matrix(1);

theta_lag_ijkl = zeros(num_rows,num_cols,num_rows,num_cols);

options = ddeset('RelTol',1e-5,'AbsTol',1e-10);
W_4_dim = W;

lag_indices(isnan(lag_indices))=0;
% Calculate solution over time
% thrid argument: [0, 150] is timespan for run
sol = dde23(@RHS_of_ODE,lag_times_reduced,initial_conditions,[0, 150],...
    options,N,w,K,num_rows,num_cols,lag_indices,num_lag_values,...
    theta_lag_ijkl, W_4_dim);
size_sol = size(sol.y)
num_timesteps = size_sol(2)
sol_matrix = mod(reshape(sol.y,num_rows,num_cols,num_timesteps),2*pi);
% plot solution
figure
for t = 1:num_timesteps
    imagesc(sol_matrix(:,:,t))
    colorbar()
    colormap hsv
    pause(0.001)
end

%-----------------------------------------------------------------------

function y_final = RHS_of_ODE(t, y, Z, N, w, K, num_rows, ...
    num_cols, lag_indices, num_lag_values,...
    theta_observed_by_ij_at_kl, W_4_dim)
% y (actual theta values) are in column form. Reshape to (i,j) matrix form:
theta_at_t = reshape(y,num_rows,num_cols);
% change from (i,j) indexing to (i,j,k,l) indexing
theta_ij_extend = repmat(theta_at_t,1,1,num_rows,num_cols);
% Z (delayed theta values) are in column form. Reshape to indices:
% (i,j,delay)
theta_lagged_values = reshape(Z,num_rows, num_cols, num_lag_values);
% create theta_observed_by_ij_at_kl (the delayed theta in sum)
for i = 1:num_rows
    for j = 1:num_cols
        for k = 1:num_rows
            for l = 1:num_cols
                if lag_indices(i,j,k,l) ~= 0
                    theta_observed_by_ij_at_kl(i,j,k,l) = ...
                        theta_lagged_values(k,l,lag_indices(i,j,k,l));
                end
            end
        end
    end
end
% Calculate subterms inside sum with indices (i,j,k,l)
sum_subterm_temp = sin(minus(theta_observed_by_ij_at_kl, theta_ij_extend));
sum_subterm = W_4_dim.*sum_subterm_temp;
% Perform sum over indices (k,l)
summed_term = sum(sum_subterm,[3,4]);

dthetadt_final = w + (K*1.0./N) .*summed_term;
y_final = reshape(dthetadt_final, num_rows*num_cols,1);


