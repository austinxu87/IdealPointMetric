%% This script gives examples on how to run learn_Md.m and alt_Mu.m
%   First, we must generate a metric, ideal point, and embedding of items
%   using generate_params.m

N = 100;            %Number of items
D = 2;              %Dimension of items/ideal point
                   
num_comps = 100;    %Number of paired comparisons to generate
frac_comps = 0.1;   %Fraction of total paired comparisons to generate

%Generate params and num_comps paired comparisons
params_num = generate_params(N, D, num_comps);  

%Generate params and frac_comps*(N choose 2) paired comparisons
params_frac = generate_params(N, D, frac_comps);

%% Perform single-step estimation with learn_Md.m

%Set optimization parameters (Fig. 2 caption)
g1 = 2;         %\Gamma_1
g2 = 0.002;     %\Gamma_2
g3 = 0.001;     %\Gamma_3
a_tik = 1;      %\alpha 

disp("-----------------------------------")
disp("-------Single Step Est.------------")
disp("-----------------------------------")

[M_l, u_l, dist_l, err] = learn_Md(params_num, a_tik, g1, g2, g3);

%Display errors
disp("Single step errors -----------------")
fprintf("UR error = %G\n", err.u_err)
fprintf("WER error = %G\n", err.Q_err)
fprintf("Fraction of top 5 = %G\n", err.topK(1))
fprintf("Fraction of top 10 = %G\n", err.topK(2))
fprintf("Fraction of top 20 = %G\n", err.topK(3))

%% Perform alternating estimation with alt_Mu.m

%Set optimization parameters (Fig. 3 caption)
g1_alt = [2; 2/3];
g2_alt = [0.002; 1/15];
g3_alt = [0.0001; 7/1500];
a_tik_alt = [1; 0.5];
niter = 100;
delta_alt = 10e-3;

disp("-----------------------------------")
disp("-------Alternating Est.------------")
disp("-----------------------------------")

[M_l_alt, u_l_alt, dist_l_alt, err_alt] = alt_Mu(params_num, niter, ...
    a_tik_alt, g1_alt, g2_alt, g3_alt, delta_alt);

%Display errors
disp("Alternating errors -----------------")
fprintf("Single step UR error = %G\n", err_alt.u_err_v(1))
fprintf("Final UR error = %G\n", err_alt.u_err_v(end))

fprintf("Single step WER error = %G\n", err_alt.Q_err_v(1))
fprintf("Final WER error = %G\n", err_alt.Q_err_v(end))

fprintf("Single step fraction of top 5 = %G\n", err_alt.topK(1,1))
fprintf("Final fraction of top 5 = %G\n", err_alt.topK(end,1))

fprintf("Single step fraction of top 10 = %G\n", err_alt.topK(1,2))
fprintf("Final fraction of top 10 = %G\n", err_alt.topK(end,2))

fprintf("Single step fraction of top 20 = %G\n", err_alt.topK(1,3))
fprintf("Final fraction of top 20 = %G\n", err_alt.topK(end,3))

