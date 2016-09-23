function [ fb ] = createFilterBank( fr )
    
    fb = zeros(220, 21);
    for i = 2:22
        s = i-1;
        p = i;
        e = i+1;
        for j = 1:220
            % fb(j,s), m = s, k = j
            if j >= fr(s) && j <= fr(p)
                fb(j,s) = (j-fr(s))/(fr(p)-fr(s));
            elseif j >= fr(p) && j <= fr(e)
                fb(j,s) = (fr(e)-j)/(fr(e)-fr(p));
            end
        end
    end
    fb = fb(:,2:21);
end

