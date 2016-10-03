function [X, varargout] = mza(Y, varargin)
% MZA Management Zone Analyst delineation algorithm MATLAB implementation.
%
%    X = MZA(Y, C) Delineates C management zones from the data Y.
%    Y is size N-by-P, where N is the number of observations
%    and P is the number of variables in an observation.
%    The output X is an N-vector of cluster assignments for each observation.
%
%    [X, FPI, NCE] = MZA(Y, [C1 ... CK]) Runs K delineations for varying C's.
%    The output X is size K-by-N, where K is the number of delineations run.
%    FPI and NCE are K-vectors of the values of the fuzziness performance
%    index and the normalized classification entropy respectively.
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
vnout = max(nargout - 1, 0);

p = inputParser();
addParam = 'addParameter';
if ~ismethod(p, addParam)
    % This makes it work in GNU Octave
    addParam = 'addParamValue';
end
p.FunctionName = 'mza';
% TODO: How to check c is integer?
p.addRequired('c', @(c) isnumeric(c) && all(c > 1) && all(c < n));
% Use MZA defaults rather than MATLAB ones
p.(addParam)('m', 1.30, ...
        @(m) isnumeric(m) && isscalar(m) && (m > 1) && isfinite(m));
p.(addParam)('d', 'euclidean'); % Give to pdist
p.(addParam)('eps', 1e-4, @(e) isnumeric(e) && isscalar(e));
p.(addParam)('max_iter', 300) % Maximum number of iterations to run
p.(addParam)('info', false);

% Parse options
p.parse(varargin{:});
c = p.Results.c;
m = p.Results.m;
d = lower(p.Results.d);
switch d
    case {'euclidean', 'mahalanobis'}
        % Do nothing
    case 'diagonal'
        %error('Diagonal-distance method not yet implemented');
        d = 'seuclidean';
    otherwise
        % pdist knows more distances than MZA, so d might still work...
        warning(['Supplied distance metric is not support by real MZA' ...
                ' (but it might still work in MATLAB)']);
end
epsilon = p.Results.eps;
lmax = p.Results.max_iter;
info = p.Results.info;

% Perform delineation(s)
X = NaN(numel(c), n);
vout = cell(numel(c), vnout);
for I = 1:numel(c)
    [X(I, :), vout{I, :}] = domza(Y, c(I), m, d, epsilon, lmax, info);
end

% Format output
for I = 1:vnout
    varargout{I} = [vout{:, I}]; %#ok
end
end

function [X, FPI, NCE, U] = domza(Y, c, m, d, epsilon, lmax, info)
% Run a single delineation for a particular value of c
    n = size(Y, 1);

    U = rand(c, n);
    U = bsxfun(@rdivide, U, sum(U, 1));
    for IT = 1:lmax
        % These two lines from stepfcm
        mf = U .^ m; % MF matrix after exponential modification
        V = mf * Y ./ (sum(mf, 2) * ones(1, size(Y, 2))); % new center

        D = pdist2(Y, V, d).';

        U_old = U;
        % These two lines from stepfcm
        tmp = D .^ (-2 / (m - 1)); % calculate new U, suppose expo != 1
        U = tmp ./ (ones(c, 1) * sum(tmp));

        % Check stopping criterion
        if norm(U - U_old) <= epsilon
            break;
        end
    end
    if info
        disp(['Stopped after ' int2str(IT) ' iterations']);
    end

    % Paper does not say how the hard assigments are made...
    [~, X] = max(U);

    % Calculate cluster performance indices
    % TODO: Should these be run on the fuzzy or the hard assignments?
    if nargout >= 2
        % Fuzziness performance index
        FPI = 1 - c / (c - 1) * (1 - mean(sum(U.^2, 1), 2));
    end
    if nargout >= 3
        % Normalized calssification entropy
        % TODO: What base for the log does MZA use?
        NCE = -mean(sum(U .* log(U), 1), 2) / (1 - c / n);
    end
end
