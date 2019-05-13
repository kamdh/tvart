clear all
close all

addpath('~/work/MATLAB/')

%% Parameters
%% test system
rng(1)

M = 20;
r = 6;
num_steps = M * 10 + 1;
num_trans = 200;
noise_std = 0.5;
noise_process = 0.0;
T = floor((num_steps - 1) / M);
noise_compensate = 0;
offset = 0;

%N_vec = [6,10,14,18,20,30,50,80,100,200,400,1000,2000,4000];
N_vec = [8000];

for N = N_vec
    %% Setup the problem
    [X, A1, A2] = switching_linear_setup(N, num_steps, noise_std, ...
                                         offset);
    save(sprintf('../data/test_data_N_%d_M_%d_sigma_%f.mat', N, num_steps, ...
                 noise_std), ...
         'X', 'A1', 'A2', 'N', 'num_steps', 'noise_std', 'offset');
end
