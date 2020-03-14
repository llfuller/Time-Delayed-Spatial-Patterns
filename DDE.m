function sol = exam1
% Using DDE by The MathWorks, Inc.

% Import values
W = readNPY(['generated_values/W.npy']);
dist_grid_arr = readNPY(['generated_values/dist_grid_arr.npy']);
phi = readNPY(['generated_values/phi.npy']);
N = readNPY(['generated_values/N.npy']);
initial_conditions = readNPY(['generated_values/initial_conditions.npy']);
scalars = readNPY(['generated_values/scalars.npy']);
lag_times_reduced = readNPY(['generated_values/lags_reduced.npy']);
lag_indices = readNPY(['generated_values/lag_indices.npy']);
w = scalars(1);
K = scalars(2);
v = scalars(3);
Omega = scalars(4);
gamma = scalars(5);
num_rows = scalars(6);
num_cols = scalars(7);
numTimeSteps = scalars(8);
size_of_lag_matrix = size(lag_times_reduced);
num_lag_values = size_of_lag_matrix(1);

theta_lag_ijkl = zeros(num_rows,num_cols,num_rows,num_cols);

extra_term = 0 % placeholder for calling dde23 with extra arguments
W_4_dim = permute(W,[2,1,4,3]);
lag_indices(isnan(lag_indices))=0;

% Calculate solution over time
% a bunch of these arguments are unnecessary but we can just leave them for
% now, I guess.
sol = dde23(@RHS_of_ODE,lag_times_reduced,initial_conditions,[0, 10],extra_term,W,dist_grid_arr,phi,N,w,K,v,Omega,gamma,...
    num_rows,num_cols,numTimeSteps,lag_indices,num_lag_values,theta_lag_ijkl,W_4_dim);
size_sol = size(sol.y)
num_timesteps = size_sol(2)
sol_matrix = mod(reshape(sol.y,num_rows,num_cols,num_timesteps),2*pi)
% plot solution
figure
for t = 1:num_timesteps
    imagesc(sol_matrix(:,:,t))
    colorbar()
    colormap hsv
    pause(0.005)
end

%-----------------------------------------------------------------------

function y_final = RHS_of_ODE(t,y,Z,~,~,~,N,w,K,~,~,...
    ~,num_rows, num_cols,~,lag_indices,~, ...
    theta_lag_ijkl,W_4_dim)

theta_at_t = permute(reshape(y,num_rows,num_cols),[2,1]);
Z_dims = size(Z);
theta_lagged_values = permute(reshape(Z,num_rows,num_cols,Z_dims(2)),[2,1,3]);
for i = 1:num_rows
    for j = 1:num_cols
        for k = 1:num_rows
            for l = 1:num_cols
                if lag_indices(i,j,k,l) ~= 0
                    theta_lag_ijkl(i,j,k,l) = theta_lagged_values(k,l,lag_indices(i,j,k,l));
                end
            end
        end
    end
end
sum_subterm = W_4_dim.*sin(theta_lag_ijkl-theta_at_t);
summed_term = nansum(sum_subterm,[3,4]);
dthetadt_final = w + (K*1.0./N) .*summed_term;
y_final = reshape(permute(dthetadt_final,[2,1]), num_rows*num_cols,1);
