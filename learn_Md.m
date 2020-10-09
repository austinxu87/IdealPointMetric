%OUTPUT(S):
%   M_l = learned Mahalanobis distance - DxD PSD Matrix
%   u_l = learned user - Dx1 vector
%   dist_l = learned distances - (num_comps)x1 vector
%   err = struct of different errors with following fields:
%       err.u_err = UR error (section 4.1) - scalar
%       err.u_E_err = Euclidean error ||emb.user - u_l||_2^2 / ||emb.user||_2^2 
%                                                   - scalar
%       err.M_err = Raw metric error ||emb.M - M_l||_F^2/||emb.M||_F^2
%                                                   -scalar
%       err.Q_err = WER error (section 4.1) - scalar
%       err.k_tau = Kendall's Tau distance between real ranking and learned
%                                           ranking -scalar
%       err.topK = fraction of top K items identified correctly for
%                                       K = 5, 10, 20 - 1x3 vector 
%       err.e = slack variables - (num_comps)x1 vector
%INPUT(S):
%   emb = emb struct generated from generate_embedding
%   a_tik = Tikhonov Regularization Parameter for recovering u_l
%   g1 = \gamma_1 (equation 8) - positive scalar
%   g2 = \gamma_2 (equation 8) - positive scalar
%   g3 = \gamma_3 (equation 8) - positive scalar
function [M_l, u_l, dist_l, err] = learn_Md(emb, a_tik, g1, g2, g3)
    %Precompute matrix which appears in constraints
    P = eye(emb.num_comps) - emb.R*pinv(emb.R);
    
    disp("Beginning CVX: Learning M.............")
    
    cvx_begin quiet
        variable M_l(emb.d,emb.d) semidefinite;
        variable dist_l(emb.n);
        variable e(emb.num_comps);
        
        minimize( ...
            sum(hinge(emb.y_k, emb.Q_k*dist_l)) + ...
            g1*sum(e) + ... 
            g2*sum(sum_square(M_l)) + ...
            g3*sum_square(dist_l) );             
        
        subject to
            e >= 0;
            P*(diag(emb.S*M_l*emb.R') - emb.Q_k*dist_l) >= -e;
            P*(diag(emb.S*M_l*emb.R') - emb.Q_k*dist_l) <= e;
            dist_l >= 0;
    cvx_end
 
    %If R*M_l is relatively ill conditioned, we add increase a_tik
    %(Two empirically found cases)
    if(cond(emb.R*M_l) > 5 && cond(emb.R*M_l) < 1000)
        a_tik = a_tik + 5*cond(emb.R*M_l);
    elseif (cond(emb.R*M_l) > 1000)
        s = svds(emb.R*M_l);
        a_tik = a_tik + 100*s(end);
    end
    
    disp("Learning u.............")
    
    %Compute u_l using regularized least squares
    u_l = 0.5*inv((emb.R*M_l)'*(emb.R*M_l) + a_tik*eye(emb.d))*...
        (emb.R*M_l)'*(diag(emb.S*M_l*emb.R') - emb.Q_k*dist_l);
    
    %Compute and store errors
    err.u_err = comp_u_err(emb.user, u_l, emb.M);
    err.u_E_err = comp_u_err(emb.user, u_l);
    err.M_err = comp_M_err(emb.M, M_l);
    err.Q_err = eig_err(emb.M, M_l);
    err.k_tau = comp_ktau(emb.distances_true, dist_l);
    err.topK = frac_top_k([5 10 20], emb.distances_true, dist_l);
    err.e = e;
    
end

%% Helper Functions
function out = hinge(y, x) 
    out = max(1-y.*x, 0);
end

function u_err = comp_u_err(u, u_l, M)
    if (nargin < 3)
        u_err = norm(u - u_l)^2/norm(u)^2;
    else
        u_err = (u - u_l)'*M*(u - u_l) / (u'*M*u);
    end
end

function M_err = comp_M_err(M, M_l)
    M_err = norm(M - M_l, 'fro')^2/norm(M, 'fro')^2;
end

function Q_err = eig_err(M, M_l)
    d = size(M,1);
    [Q, L] = eig(M);
    [Q_l, ~] = eig(M_l);
    
    %Check if emb.M is identity or not. Compute WER error accordingly
    if(sum(sum(M - eye(d))) < eps)
        QtQ = zeros(d,d);
        for i = 1:d
            QtQ(i,i) = max(abs(Q_l(:,i)));
        end
        Q_err = norm(L .* QtQ - L, 'fro')/sqrt(d);
    else
        Q_err = norm(L .* abs(Q'*Q_l) - L,'fro')/norm(L, 'fro');
    end
end


function ktau = comp_ktau(distances_true, dist_l)
    [~, true_rank] = sort(distances_true);
    [~, learned_rank] = sort(dist_l);
    
    ktau = 0;
    for i = 1:length(distances_true)
        for j = i+1:length(distances_true)
            ktau = ktau + max(-sign((true_rank(i) - true_rank(j))*(learned_rank(i) - learned_rank(j))), 0);
        end
    end
    
    ktau = ktau / nchoosek(length(distances_true), 2);
end

function frac = frac_top_k(k, distances_true, dist_l)
    [~, distances_true] = sort(distances_true);
    [~, learned_rank] = sort(dist_l);
    
    %k is a vector of the top k values to compute fraction correct of
    frac = zeros(size(k));
    for i = 1:length(k)
        frac(i) = length(intersect(learned_rank(1:k(i)), distances_true(1:k(i))))/k(i);
    end
end