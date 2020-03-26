function sol = Oscillators2D
% Using DDE by The MathWorks, Inc.
% See DDE tutorial pdf.
addpath('npy-matlab')
addpath('generated_values')
% Import values
W = single(readNPY(['generated_values/W.npy']));
%dist_grid_arr = readNPY(['generated_values/dist_grid_arr.npy']);
%phi = readNPY(['generated_values/phi.npy']);
N = single(readNPY(['generated_values/N.npy']));
initial_conditions = single(readNPY(['generated_values/initial_conditions.npy']));
scalars = readNPY(['generated_values/scalars.npy']);
lag_times_reduced = single(readNPY(['generated_values/lags_reduced.npy']));
lag_indices = single(readNPY(['generated_values/lag_indices.npy']));
w = single(scalars(1));
K = single(scalars(2));
%v = scalars(3);
%Omega = scalars(4);
%gamma = scalars(5);
num_rows = scalars(6);
num_cols = scalars(7);
%numTimeSteps = scalars(8);
size_of_lag_matrix = size(lag_times_reduced);
num_lag_values = size_of_lag_matrix(1);

theta_lag_ijkl = single(zeros(num_rows,num_cols,num_rows,num_cols));

options = ddeset('RelTol',1e-5,'AbsTol',1e-10);
W_4_dim = W;

lag_indices(isnan(lag_indices))=1;

% Calculate solution over time
% thrid argument: [0, 150] is timespan for run
theta_observed_by_ij_at_kl_temp = single(zeros(num_rows*num_cols, num_rows, num_cols));
tic
sol = dde23(@RHS_of_ODE,lag_times_reduced,initial_conditions,[0, 500],...
    options,N,w,K,num_rows,num_cols,lag_indices,num_lag_values, W_4_dim, ...
    theta_observed_by_ij_at_kl_temp);
toc
size_sol = size(sol.y)
num_timesteps = size_sol(2)
sol_matrix = mod(reshape(sol.y,num_rows,num_cols,num_timesteps),2*pi);
save 'sol_matrix.mat' sol_matrix
save 'sol_object.mat' sol
% plot solution
yint = deval(sol, linspace(0, 500))
size_yint = size(yint)
plot_matrix_temp = reshape(yint,num_rows,num_cols,size_yint(2))
plot_matrix = mod(plot_matrix_temp,2*pi);
figure
for t = 1:num_timesteps 
    imagesc(plot_matrix(:,:,t))
    colorbar()
    colormap hsv
    pause(0.1)
end

%-----------------------------------------------------------------------

function y_final = RHS_of_ODE(t, y, Z, N, w, K, num_rows, ...
    num_cols, lag_indices, num_lag_values, W_4_dim,...
    theta_observed_by_ij_at_kl_temp)
% y (actual theta values) are in column form. Reshape to (i,j) matrix form:
theta_at_t = single(reshape(y,num_rows,num_cols));
% change from (i,j) indexing to (i,j,k,l) indexing
theta_ij_extend = repmat(theta_at_t,1,1,num_rows,num_cols);
% Z (delayed theta values) are in column form. Reshape to indices (i,j,delay)
theta_lagged_values = reshape(single(Z),num_rows, num_cols, num_lag_values);
for k = 1:num_rows
    for l = 1:num_cols
        theta_observed_by_ij_at_kl_temp(:,k,l) = theta_lagged_values(k,l,lag_indices(:,:,k,l));
    end
end
theta_observed_by_ij_at_kl = single(reshape(theta_observed_by_ij_at_kl_temp, num_rows, num_cols, num_rows, num_cols));

% Calculate subterms inside sum with indices (i,j,k,l)
sum_subterm_temp = sin(minus(theta_observed_by_ij_at_kl, theta_ij_extend));
sum_subterm = W_4_dim.*sum_subterm_temp;
% Perform sum over indices (k,l)
summed_term = sum(sum_subterm,[3,4]);

dthetadt_final = w + (K*1.0./N) .*summed_term;
y_final = reshape(dthetadt_final, num_rows*num_cols,1);


