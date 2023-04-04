% Define constraints function
function [c, ceq] = portfolio_constraints(weights)
    c = sum(weights) - 1; % sum of weights must equal 1
    ceq = [];
end