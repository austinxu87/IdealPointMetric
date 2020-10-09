%OUTPUT(S): 
%   emb - Struct with following fields:
%       emb.D = Dimension of embedding (repeated from input) - integer
%       emb.N = Number of items (repeated from input) - integer
%       emb.M = Mahalanobis metric - DxD PSD matrix
%       emb.items = items - DxN matrix of items, with x_i = items(:,i)
%       emb.user = true ideal point - Dx1 vector
%       emb.num_comps = Number of paired comparisons generated
%       emb.y_k = outcome of paired comparisons (+/-1) - (num_comps)x1
%                                                           vector
%       emb.distances_true = distances between x_i and user under M
%                                   - Nx1 vector
%       emb.diff_norm = difference in Euclidean distances (for comparison)
%                                   - (num_comps)x1 vector
%       emb.Q_k = Q_{\Gamma} matrix from paper - (num_comps)x(num_comps)
%                                                   matrix
%       emb.R = R matrix from paper - (num_comps)x(num_comps) matrix
%       emb.S = S matrix from paper - (num_comps)x(num_comps) matrix
%INPUT(S):
%   N = Number of items - integer
%   D = Dimension of embedding - integer
%   comps_num = controls number of comparisons. Two options:
%               -decimal between 0 and 1: Fraction of total possible
%                paired comparisons to generate
%               -integer between 1 and (N choose 2): Number of paired  
%                comparisons to generate
function emb = generate_params(N, D, comps_num)
    %Thresholds to ensure M is conditioned well enough and that u is
    %identifiable. Described in section 4.1
    eps_f = 0.5;
    eps_s = 0.25;
    eps_p = 0.2;

    num_pairs = nchoosek(N,2);
    
    %% Generate M, L, user, items
    L = randn(D,D);
    M = L'*L;

    s = svds(M);
    
    %Generate M until thresholds are met
    while(norm(M, 'fro') < eps_f || s(end) < eps_s)
        L = randn(D,D);
        M = L'*L;
        s = svds(M);
    end
    
    %Generate user until thresholds met
    user = 2*rand(D,1) - 1;
    while(norm(M*user)/norm(user) < eps_p)
        user = 2*rand(D,1) - 1;
    end
    
    %Generate items
    items = 4*rand(D, N) - 2;

    %% Compute all true distances for items
    distances_true = zeros(N,1);
    for i = 1:N
        distances_true(i) = ((items(:,i) - user)'*M)*(items(:,i) - user);
    end

    %% Generate comparisons
    %Case 1: comps_num is fraction of paired comparisons to generate
    if(comps_num < 1 && comps_num > 0)
        num_comps = round(comps_num*num_pairs);
    %Case 2: comps_num is number of paired comparisons to generate
    elseif (comps_num > 1 && comps_num <= num_pairs)
        num_comps = comps_num;
    end
    
    %Generate Q_k, R, S
    inds_lin = randperm(num_pairs, num_comps)';
    inds_sub = lin2sub(inds_lin, N);   
    Q_k = sparse([(1:num_comps)'; (1:num_comps)'],...
       [inds_sub(1:num_comps,1); inds_sub(1:num_comps, 2)],...
       [ones(num_comps,1); -1*ones(num_comps,1)], num_comps, N);
        
    R = (items(:,inds_sub(:,1)) - items(:,inds_sub(:,2)))';
    S = (items(:,inds_sub(:,1)) + items(:,inds_sub(:,2)))';
    
    %Generate paired comparisons without noise
    y_k = sign(Q_k*distances_true);
    
    %Generate differences in Euclidean distances between items
    %(For comparison with "Lost Without a Compass" paper algorithm
    diff_norm = zeros(num_comps,1);
    for i = 1:num_comps
        diff_norm(i) = norm(items(:,inds_sub(i,1)))^2 ...
            - norm(items(:,inds_sub(i,2)))^2;
    end

    %% Store outputs
    emb.d = D;
    emb.n = N;
    
    emb.M = M;
    emb.items = items;
    emb.user = user;
    
    emb.num_pairs = num_pairs;
    emb.num_comps = num_comps;
    
    emb.distances_true = distances_true;
    emb.diff_norm = diff_norm;
    
    emb.y_k = y_k;
    emb.Q_k = Q_k;
    emb.R = R;
    emb.S = S;
    
end

%% Helper functions
%Convert linear indices to matrix subscripts.
%   Adapted from: https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix/27088560
function z = lin2sub(ind, n)
  rvLinear = (n*(n-1))/2 - ind;
  k = floor((sqrt(1+8.*rvLinear)-1)/2);

  j= rvLinear - k.*(k+1)./2;

  z=[n-(k+1), n - j];
end
