% Define objective function
function f = NSGA(weights, rets, rf, sd, target_return)
    portfolio_return = weights * rets;
    portfolio_sd = sqrt(weights' * sd * weights);
    sharpe_ratio = (portfolio_return - rf) / portfolio_sd;
    deviation_from_target = abs(portfolio_return - target_return);
    f = [-sharpe_ratio, deviation_from_target];
end

% Define constraints function
function [c, ceq] = portfolio_constraints(weights)
    c = sum(weights) - 1; % sum of weights must equal 1
    ceq = [];
end

% Example usage
tickers = {'AAPL', 'MSFT', 'GOOG', 'AMZN'};
start_date = '2018-01-01';
end_date = '2020-12-31';

% Get stock prices
conn = yahoo('http://download.finance.yahoo.com');
prices = cellfun(@(t) fetch(yahoo,t,start_date,end_date), tickers, 'UniformOutput', false);
prices = cellfun(@(p) p(:, 5), prices, 'UniformOutput', false);

% Calculate returns
rets = cellfun(@(p) tick2ret(p, 'continuous'), prices, 'UniformOutput', false);
rets = cell2mat(rets);

% Calculate mean and covariance
mu = mean(rets)';
sigma = cov(rets);

% Set risk-free rate and target return
rf = 0.02;
target_return = 0.08;

% Define NSGA-II parameters
nsga2_params = gaoptimset('PopulationSize', 100, ...
                          'Generations', 100, ...
                          'CrossoverFraction', 0.7, ...
                          'EliteCount', 2, ...
                          'MutationFcn', {@mutationuniform, 0.1});

% Define bounds and initial population
n_assets = length(mu);
bounds = repmat([0, 1], n_assets, 1);
initial_population = repmat(1/n_assets, 100, n_assets);

% Optimize portfolio
[weights, ~, ~, ~] = gamultiobj(@(w)portfolio_objective_function(w, mu, rf, sigma, target_return), ...
                                 n_assets, [], [], [], [], ...
                                 bounds(:,1), bounds(:,2), ...
                                 @(w)portfolio_constraints(w), ...
                                 nsga2_params);
