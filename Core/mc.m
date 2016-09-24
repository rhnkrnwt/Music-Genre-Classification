function [ y ] = mc( x )
    % convert from Hertz to Mel scale units
    y  = 1125 * log(1 + x/700);
end

