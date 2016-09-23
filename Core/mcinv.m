function [ y ] = mcinv(x)
    y = 700 .* (exp(x./1125) - 1);
    y = [0; y];
end

