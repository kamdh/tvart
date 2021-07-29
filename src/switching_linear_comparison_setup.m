clear all
close all

addpath('~/work/MATLAB/')

%% Parameters
%% test system
rng(1)

M = 20;
num_steps = M * 10 + 1;
%num_steps = M * 100 + 1;
num_trans = 200;
noise_std = 0.5;
noise_process = 0.0;
T = floor((num_steps - 1) / M);
noise_compensate = 0;
offset = 0;
num_reps = 3;

%% for N sweeps
%N_vec = [6,10,14,18,20,30,50,80,100,200,400,1000,2000,4000];
%noise_vec = [0.5];

%% for noise_std sweeps
N_vec = [10,20,40,80,160,320]; %,2000,4000];
noise_vec = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.];

%N_vec = [8000];
for noise_std = noise_vec
    for N = N_vec
        for rep = 1:num_reps
            %% Setup the problem
            [X, A1, A2] = switching_linear_setup(N, num_steps, noise_std, ...
                                                 offset);
            save(sprintf('../data/test_data_N_%d_M_%d_sigma_%f_rep_%d.mat', N, num_steps, ...
                         noise_std, rep), ...
                 'X', 'A1', 'A2', 'N', 'num_steps', 'noise_std', 'offset');
        end
    end
end