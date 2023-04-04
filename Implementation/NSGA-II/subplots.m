% Define objective function
function f = portfolio_objective_function(weights, rets, rf, sd, target_return)
    portfolio_return = weights * rets;
    portfolio_sd = sqrt(weights' * sd * weights);
    sharpe_ratio = (portfolio_return - rf) / portfolio_sd;
    deviation_from_target = abs(portfolio_return - target_return);
    f = [-sharpe_ratio, deviation_from_target];
end