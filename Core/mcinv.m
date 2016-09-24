function [ y ] = mcinv(x)
    % convert from Mel scale units to Hertz
    y = 700 .* (exp(x./1125) - 1);
    y = [0; y];
end

