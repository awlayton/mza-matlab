function [C, U, FPI, NCE] = mza(Y, varargin)
% MZA Management Zone Analyst delineation algorithm MATLAB implementation.
%
%    X = MZA(Y, C) Delineates C management zones from the data Y.
%    Y is size N-by-P, where N is the number of observations
%    and P is the number of variables in an observation.
%    The output X is an N-vector of cluster assignments for each observation.
%
%    X = MZA(Y,  ..., 'Param1', val1, ...) enables you to specify algorithm
%    parameter name/value pairs.  Parameters are:
%
%       'm' -- fuzziness exponent for fuzzy c-means (default 1.30)
%
%       'd' -- distance metric (default 'euclidean'), can be one of:
%         -  'euclidean'
%         -  'mahalanobis'
%         -  'diagnoal'
%
%       'eps' -- epsilon value to use for stopping criterion (default 1e-4)
%
%       'max_iter' -- maximum number of iterations to run (default 300)
%
% References:
%    MZA Paper http://handle.nal.usda.gov/10113/8380
%
% Author:
%    Alex Layton <alex@layton.in> (http://alex.layton.in)

% Number of observations
n = size(Y, 1);

p = inputParser();
p.FunctionName = 'mza';
% TODO: How to check c is integer?
p.addRequired('c', @(c) isnumeric(c) && isscalar(c) && (c > 1) && (c < n));
% Use MZA defaults rather than MATLAB ones
p.addOptional('m', 1.30, ...
        @(m) isnumeric(m) && isscalar(m) && (m > 1) && isfinite(m));
p.addOptional('d', 'euclidean'); % Give to pdist
p.addOptional('eps', 1e-4, @(e) isnumeric(e) && isscalar(e));
p.addOptional('max_iter', 300) % Maximum number of iterations to run

p.parse(varargin{:});

c = p.Results.c;

d = lower(p.Results.d);
switch d
    case {'euclidean', 'mahalanobis'}
        % Do nothing
    case 'diagonal'
        %error('Diagonal-distance method not yet implemented');
        d = 'seuclidean'; % Same thing?
    otherwise
        % pdist knows more distances than MZA, so d might still work...
        warning(['Supplied distance metric is not support by real MZA' ...
                ' (but it might still work in MATLAB)']);
end

U = initfcm(c, n);
for I = 1:p.Results.max_iter
    % These two lines from stepfcm
    mf = U.^p.Results.m;       % MF matrix after exponential modification
    V = mf*Y./(sum(mf,2)*ones(1,size(Y,2))); %new center

    D = pdist2(Y, V, d).';

    U_old = U;
    % These two lines from stepfcm
    tmp = D.^(-2/(p.Results.m-1));      % calculate new U, suppose expo != 1
    U = tmp./(ones(c, 1)*sum(tmp));

    % Check stopping criterion
    if norm(U - U_old) <= p.Results.eps
        break;
    end
end

% Paper does not say how the hard assigments are made...
[~, C] = max(U);

% Calculate cluster performance indices
% TODO: Should these be run on the fuzzy or the hard assignments?
if nargout >= 3
    % Fuzziness performance index
    FPI = 1 - c / (c - 1) * (1 - mean(sum(U.^2, 1), 2));
end
if nargout >= 4
    % Normalized calssification entropy
    % TODO: What base for the log does MZA use?
    NCE = -mean(sum(U .* log(U), 1), 2) / (1 - c / n);
end

end
