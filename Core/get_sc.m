function [ sc ] = get_sc( P )
den = sum(P');
u = [1:size(P,1)];
num = u .* den;
sc = sum(num) / sum(den);
end

