clear all
close all

addpath('~/work/MATLAB/')

%% Parameters
%% test system
rng(1)

M = 1;
r = 6;
num_steps = M * 160 + 1;
num_trans = 200;
noise_std = 0.2;
noise_process = 0.0;
T = floor((num_steps - 1) / M);
noise_compensate = 0;
offset = 0;

N_vec = [6,12,24,50,100,200,400,1000,2000,4000];
%N_vec = [8000];

for N = N_vec
    %% Setup the problem
    [X, thetas, U] = smooth_linear_setup(N, num_steps, noise_std);
    save(sprintf('../data/test_data_smooth_N_%d_M_%d_sigma_%f.mat', N, num_steps, ...
                 noise_std), ...
         'X', 'thetas', 'U', 'N', 'num_steps', 'noise_std');
end
