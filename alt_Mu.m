%OUTPUT(S):
%   M_l = final learned Mahalanobis distance at end iter - DxD PSD Matrix
%   u_l = final learned user at end iter - Dx1 vector
%   dist_l_out = final learned distances at end iter - (num_comps)x1 vector
%   err = struct of different errors with following fields:
%       err.u_err_v = UR error vector (section 4.1) - final iter x 1 vector
%       err.M_err_v = Raw metric error ||emb.M - M_l||_F^2/||emb.M||_F^2
%                                   - final iter x 1 vector
%       err.Q_err_v = WER error (section 4.1) - (niter+1)x3 vector
%       err.k_tau_v = Kendall's Tau distance between real ranking and learned
%                                   ranking - final iter x 1 vector
%       err.topK = fraction of top K items identified correctly for
%                                   K = 5, 10, 20 - final iter x 3 vector 
%INPUT(S):
%   emb = emb struct generated from generate_embedding
%   niter = number of iterations - positive integer
%   a_tik = Tikhonov Regularization Parameter for recovering u_l 
%                   - 2x1 or (niter+1)x1 vector
%   g1 = \gamma_1 (equation 8) - 2x1 or (niter+1)x1 vector
%   g2 = \gamma_2 (equation 8) - 2x1 or (niter+1)x1 vector
%   g3 = \gamma_3 (equation 8) - 2x1 or (niter+1)x1 vector
%   delta = stopping criteria for iterations - positive scalar
function [M_l, u_l, dist_l_out, err] = alt_Mu(emb, niter, a_tik_in, ...
    g1, g2, g3, delta)
    
    assert(numel(a_tik_in) == niter+1 || numel(a_tik_in) == 2, ...
        "Number of a_tik params must be 2 or niter")
    
    assert(numel(g1) == niter+1 || numel(g1) == 2, ...
        "Number of gamma 1 params must be 2 or niter+1")

    assert(numel(g2) == niter+1 || numel(g2) == 2, ...
        "Number of gamma 1 params must be 2 or niter+1")
    
    assert(numel(g3) == niter+1 || numel(g3) == 2, ...
        "Number of gamma 1 params must be 2 or niter+1")
    
    %Precompute matrix which appears in constraints
    P = eye(emb.num_comps) - emb.R*pinv(emb.R);
    
    %% Initial estimation of M and u
    disp("Single step: Learning M.............")
    cvx_begin quiet
        variable M_l(emb.d, emb.d) semidefinite;
        variable dist_l(emb.n);
        variable e(emb.num_comps);

        minimize( ...
            sum(hinge(emb.y_k, emb.Q_k*dist_l)) + ...
            g1(1)*sum(e) + ... 
            g2(1)*sum(sum_square(M_l)) + ...
            g3(1)*sum_square(dist_l) );           
        
        subject to
            e >= 0;
            P*(diag(emb.S*M_l*emb.R') - emb.Q_k*dist_l) >= -e;
            P*(diag(emb.S*M_l*emb.R') - emb.Q_k*dist_l) <= e;
            dist_l >= 0;
    cvx_end
    
    disp("Single step: Learning u.............")
    
    a_tik = a_tik_in(1);
    if(cond(emb.R*M_l) > 5 && cond(emb.R*M_l) < 1000)
        a_tik = a_tik + 5*cond(emb.R*M_l);
    elseif (cond(emb.R*M_l) > 1000)
        s = svds(emb.R*M_l);
        a_tik = a_tik + 100*s(end);
    end
    u_l = 0.5*inv((emb.R*M_l)'*(emb.R*M_l) + a_tik*eye(emb.d))*(emb.R*M_l)'...
        *(diag(emb.S*M_l*emb.R') - emb.Q_k*dist_l);
    
    dist_l_out(:,1) = dist_l;
    
    disp('Single step estimate complete, beginning iterations.............')
    %% Set alternating parameters
    
    %Check if fixed parameters for all iterations or adaptive
    if(length(a_tik_in) == 2)
        a_tik_in(end+1:niter+1) = a_tik_in(2);
    end
    
    if(length(g1) == 2)
        g1(end+1:niter+1) = g1(2);
    end
    
    if(length(g2) == 2)
        g2(end+1:niter+1) = g2(2);
    end
    
    if(length(g3) == 2)
        g3(end+1:niter+1) = g3(2);
    end
    
    %% Initialize iteration outputs
    kt_v = zeros(niter+1,1);
    kt_v(1) = comp_ktau(emb.distances_true, emb.items, M_l, u_l);
    
    u_err_v = zeros(niter+1,1);
    u_err_v(1) = comp_u_err(emb.user, u_l, emb.M);
    
    M_err_v = zeros(niter+1,1);
    M_err_v(1) = comp_M_err(emb.M, M_l);
    
    Q_err_v = zeros(niter+1,1);
    Q_err_v(1) = eig_err(emb.M,M_l);
    
    top_K_v = zeros(niter+1,3);
    top_K_v(1,:) = frac_top_k([5 10 20], emb.distances_true, dist_l);
    %% Begin alternating steps
    for iter = 2:niter+1
        if(mod(iter, 10) == 0)
            disp(strcat('Iter =  ', num2str(iter)))
        end
        
        cvx_begin quiet
        variable M_l(emb.d, emb.d) semidefinite;
        variable dist_l(emb.n);
        variable e(emb.num_comps);

        minimize( ...
            sum(hinge(emb.y_k, emb.Q_k*dist_l)) + ...
            g1(iter)*sum(e) + ... 
            g2(iter)*sum(sum_square(M_l)) + ...
            g3(iter)*sum_square(dist_l) );              
        
        subject to
            e >= 0;
            diag(emb.S*M_l*emb.R') - emb.Q_k*dist_l - 2*emb.R*M_l*u_l >= -e;
            diag(emb.S*M_l*emb.R') - emb.Q_k*dist_l - 2*emb.R*M_l*u_l <= e;
            dist_l >= 0;
        cvx_end
        
        a_tik = a_tik_in(iter);
        if(cond(emb.R*M_l) > 5 && cond(emb.R*M_l) < 1000)
            a_tik = a_tik + 5*cond(emb.R*M_l);
        elseif (cond(emb.R*M_l) > 1000)
            s = svds(emb.R*M_l);
            a_tik = a_tik + 100*s(end);
        end
        
        u_l = 0.5*inv((emb.R*M_l)'*(emb.R*M_l) + a_tik*eye(emb.d))*(emb.R*M_l)'*(diag(emb.S*M_l*emb.R') - emb.Q_k*dist_l);
        
        u_err_v(iter) = comp_u_err(emb.user, u_l, emb.M);
        M_err_v(iter) = comp_M_err(emb.M, M_l);
        Q_err_v(iter) = eig_err(emb.M,M_l);
        kt_v(iter) = comp_ktau(emb.distances_true, emb.items, M_l, u_l);
        top_K_v(iter,:) = frac_top_k([5 10 20], emb.distances_true, dist_l);
        
        %If UR error between two successive iterations < delta and 
        %the UR error at iter - 1 > UR error at iter, break 
        if(u_err_v(iter-1) - u_err_v(iter) < delta && u_err_v(iter-1) - u_err_v(iter) >= 0)
            break;
        
        %If the user error increases, return the previous iter output and break
        elseif (u_err_v(iter-1) - u_err_v(iter) < 0)
            u_err_v(iter) = 0;
            iter = iter - 1;
            break;
        end
               
    end
    
    dist_l_out = dist_l;
    err.u_err_v = u_err_v(1:iter);
    err.M_err_v = M_err_v(1:iter);
    err.Q_err_v = Q_err_v(1:iter);
    err.k_tau_v = kt_v(1:iter);
    err.topK = top_K_v(1:iter,:);
end

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

function ktau = comp_ktau(distances_true, items, M_l, u_l)
    dist_l = diag((items - u_l)'*M_l*(items - u_l));

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

%       a_tik = 0.5
%     minimize( ...
%     sum(sum_square(M_l)) + ...
%     10*sum(e) + ...
%     15*sum(hinge(emb.y_k, emb.Q_k*dist_l)) + ...
%     0.07*sum_square(dist_l) );   