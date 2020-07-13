function [xq, lFapp]=tt_irt(xsf, f, q)
% Inverse CDF (Rosenblatt) transform through linear splines of density F
% This version computes CDF inverses in each variable via quadratic splines
% Inputs (sizes):
%   xsf: (d x 1) cell array of grid points (inc. boundaries) for all dimensions
%   f: TT format of the PDF (tt_tensor), computed on a grid given in xsf
%   q: seed points from [0,1]^d, defining the samples (an M x d matrix)
%
% Outputs (sizes):
%   xq: samples mapped from q by the inverse CDF (M x d matrix)
%   lFapp: log(approximate PDF) (inv. Jacobian of xq) sampled on q


if any(q(:)<-1e-12)||any(q(:)>1+1e-12)
    keyboard;
end

if (isa(f, 'tt_tensor'))
    f = core2cell(f);
end
% Parse TT: extract dimensions and ranks
d = numel(f);
n = cellfun(@(x)size(x,2), f);
rf = [1; cellfun(@(x)size(x,3), f)];

% Check grid points
if (isa(xsf, 'cell'))
    xsf = cell2mat(xsf);
end
if (size(xsf,1)~=sum(n))
    error('number of grid points in xsf should be sum of mode sizes in f');
end
pos = cumsum([1; n]); % positions where each x grid vector starts

% Prepare integrated right TT blocks (P array)
P = cell(d,1);
Pprev = 1;
h = cell(d,1); % array of grid intervals
for k=d:-1:1
    % Grid intervals
    h{k} = zeros(1,n(k));
    h{k}(2:n(k)) = xsf(pos(k)+1:pos(k)+n(k)-1) - xsf(pos(k):pos(k)+n(k)-2);    
    % Prepare semi-marginal f^{(k)}*C{k}, to be integrated into a CDF    
    P{k} = reshape(f{k}, rf(k)*n(k), rf(k+1));
    P{k} = P{k}*Pprev;
    P{k} = reshape(P{k}, rf(k), n(k));
    f{k} = permute(f{k}, [1,3,2]); % this is needed for tracemult conditioning
    if (k>1)
        % integrate via trapezoidal rule, equiv to
        % Linear-spline integration of (k..d)-th TT blocks
        % It's faster using a sparse matrix
        S = spdiags(0.5*ones(n(k),2), 0:1, n(k), n(k));
        Pprev = P{k}*S;
        Pprev = Pprev.*repmat(h{k}, rf(k), 1);
        Pprev = sum(Pprev, 2);
    end
end
% Now each P{k} = \int f^{(>=k)}(x_k...x_d) dx_{>k}

% Storage for x and approx. density
M = size(q,1);
xq = zeros(M,d);
if (nargout>1)
    lFapp = zeros(M,1);
end

% Iterate forward, computing conditional PDFs
% Prevent OOM by blocking
Mb = 2^16;
num_blocks = ceil(M/Mb);
for i_block=1:num_blocks
    % Determine the position of the current block
    start_pos = (i_block-1)*Mb + 1;
    if (i_block*Mb>M)
        Mb = M - start_pos + 1;
    end
    fkm1 = ones(1, Mb); % This will store conditioned left TT blocks f^{(<k)}(x_{<k}^*)
    
    for k=1:d
        %%% Prepare 1D PDF and CDF
        % Condition on left variables -- multiply with sampled left TT blocks
        fk = fkm1.'*P{k};
        % make this nonnegative.
        fk = abs(fk);
        % Integrate up to whole points using sparse mv
        S = spdiags(0.5*ones(n(k),2), 0:1, n(k), n(k));
        Ck = fk*S;
        Ck = Ck.*repmat(h{k}, Mb, 1);       
        Ck = cumsum(Ck, 2);

        % normalize
        Cmax = Ck(:,n(k));
        ind_zero = find(Cmax==0);
        % Eliminate possible exact zeros
        if (~isempty(ind_zero))
            fk(ind_zero, :) = 1;
            Ck(ind_zero, :) = repmat(cumsum(h{k}, 2), numel(ind_zero), 1);
            Cmax(ind_zero) = Ck(ind_zero,n(k));
        end
        Cmax = repmat(Cmax, 1, n(k));
        Ck = Ck./Cmax;
        fk = fk./Cmax;
        
        %%% Invert the marginal CDF Ck
        % Binary search for the closest to q points
        qk = q(start_pos:(start_pos+Mb-1),k);
        i0 = ones(Mb,1);
        i2 = n(k)*ones(Mb,1);
        while (any((i2-i0)>1))
            i1 = floor((i0+i2)/2);
            C1 = tracemult(Ck, i1); % MEXed extraction C1(i) = Ck(i,i1(i));
            ind_left = qk>C1; % at these indices i0 becomes i1
            i0(ind_left) = i1(ind_left);
            i2(~ind_left) = i1(~ind_left);
        end
        
        % Sample C and f onto i0:i0+1
        C1 = tracemult(Ck,i0);      % C1(i) = Ck(i,i0(i));
        C2 = tracemult(Ck, i0+1);   % C2(i) = Ck(i,i0(i)+1);
        f1 = tracemult(fk,i0);      % f1(i) = fk(i,i0(i));
        f2 = tracemult(fk, i0+1);   % f1(i) = fk(i,i0(i)+1);
        % We can miss a point, check that
        ind_missed = find((C1>(qk+1e-12)) | (C2<(qk-1e-12)));
        if (numel(ind_missed)>0)
            keyboard;
        end
        % Solve the _quadratic_ equation, since C is an integrand of linear
        % interpolant f
        x1 = xsf(pos(k)+i0-1);  % grid points at i0, i0+1
        x2 = xsf(pos(k)+i0);
        h3 = x2 - x1; % grid intervals
        
        Aq = 0.5*(f2-f1)./h3;
        Dq = f1.^2 + 4*Aq.*(qk-C1);
        xk = x1 + (-f1 + sqrt(abs(Dq)))./(2*Aq);
        ind_missed = find(Aq==0); % At these indices we need to solve a linear equation
        xk(ind_missed) = x1(ind_missed) + (qk(ind_missed)-C1(ind_missed))./f1(ind_missed);

        % Check if we are outside the range
        ind_missed = find(xk>x2);
%         if (numel(ind_missed)>0)&&any(xk(ind_missed)-x2(ind_missed)>1e-10)
%             keyboard;
%         end
        xk(ind_missed) = x2(ind_missed);
        ind_missed = find(xk<x1);
%         if (numel(ind_missed)>0)&&any(x1(ind_missed)-xk(ind_missed)>1e-10)
%             keyboard;
%         end        
        xk(ind_missed) = x1(ind_missed);
                        
        xq(start_pos:(start_pos+Mb-1),k) = xk;
        
        % Linear interpolation coefficients into xk
        Aq = (x2-xk)./h3;
        Bq = (xk-x1)./h3;
        if (nargout>1)
            % Interpolate fk into xk:
            % this is our actual conditional probability at the k-th step
            % The whole density is a product of those
            fk = f1.*Aq + f2.*Bq;
            lFapp(start_pos:(start_pos+Mb-1)) = lFapp(start_pos:(start_pos+Mb-1)) + log(fk);
        end
        
        if (k<d)
            % Sample the k-th block onto xk via linear interpolation
            % On-the-fly product
            % fkm1(i,:)*fk(:,i0(i),:)*Aq(i) + fkm1(i,:)*fk(:,i0(i)+1,:)*Bq(i)
            fkm1 = reshape(fkm1, 1, rf(k), Mb);
            Aq = reshape(Aq, 1, 1, Mb);
            Bq = reshape(Bq, 1, 1, Mb);
            fkm1 = tracemult(fkm1, i0, f{k}).*Aq + tracemult(fkm1, i0+1, f{k}).*Bq;
            fkm1 = reshape(fkm1, rf(k+1), Mb);
        end
    end
end
end
